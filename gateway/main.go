package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/streadway/amqp"
)

type Task struct {
	ID        string    `json:"id"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	Result    any       `json:"result,omitempty"`
	Error     string    `json:"error,omitempty"`
}

var (
	tasks   = make(map[string]*Task)
	rdb     *redis.Client
	db      *sql.DB
	channel *amqp.Channel
	queue   amqp.Queue

	// Метрики Prometheus
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint"},
	)
	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)
)

func init() {
	// Регистрируем метрики
	prometheus.MustRegister(httpRequestsTotal)
	prometheus.MustRegister(httpRequestDuration)

	// Redis
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://localhost:6379/0"
	}
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		log.Fatalf("Invalid REDIS_URL: %v", err)
	}
	rdb = redis.NewClient(opt)

	// PostgreSQL
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://user:pass@localhost:5432/prices?sslmode=disable"
	}
	db, err = sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("Failed to connect to DB: %v", err)
	}
	if err = db.Ping(); err != nil {
		log.Fatalf("DB ping failed: %v", err)
	}

	// RabbitMQ
	rabbitURL := os.Getenv("RABBITMQ_URL")
	if rabbitURL == "" {
		rabbitURL = "amqp://guest:guest@localhost:5672/"
	}
	_, ch := connectRabbitMQ(rabbitURL)
	channel = ch
	queue, err = channel.QueueDeclare("tasks", true, false, false, false, nil)
	if err != nil {
		log.Fatalf("Failed to declare queue: %v", err)
	}
}

func connectRabbitMQ(url string) (*amqp.Connection, *amqp.Channel) {
	var conn *amqp.Connection
	var err error
	for i := 1; i <= 10; i++ {
		conn, err = amqp.Dial(url)
		if err == nil {
			break
		}
		log.Printf("Waiting for RabbitMQ... (%d/10): %v", i, err)
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ after 10 attempts: %v", err)
	}
	ch, err := conn.Channel()
	if err != nil {
		log.Fatalf("Failed to open channel: %v", err)
	}
	return conn, ch
}

// middleware для сбора метрик
func metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		// Оборачиваем ResponseWriter, чтобы захватить код ответа
		ww := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(ww, r)
		duration := time.Since(start).Seconds()

		// Инкрементируем счётчик и наблюдаем длительность
		httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path).Inc()
		httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
	})
}

// responseWriter для перехвата статус-кода
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func main() {
	r := mux.NewRouter()

	// Логирование и метрики
	r.Use(loggingMiddleware)
	r.Use(metricsMiddleware)

	// Метрики Prometheus
	r.Handle("/metrics", promhttp.Handler())

	// Healthcheck
	r.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}).Methods("GET")

	// API
	r.HandleFunc("/upload", uploadHandler).Methods("POST")
	r.HandleFunc("/task/{id}", taskHandler).Methods("GET")

	log.Println("Gateway listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("→ %s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
		log.Printf("← %s %s [%v]", r.Method, r.URL.Path, time.Since(start))
	})
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("file")
	if err != nil {
		log.Printf("ERROR upload: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		log.Printf("ERROR read file: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	taskID := uuid.New().String()
	task := &Task{ID: taskID, Status: "pending", CreatedAt: time.Now()}
	tasks[taskID] = task

	log.Printf("Creating task %s, size=%d bytes", taskID, len(data))

	err = channel.Publish(
		"", queue.Name, false, false,
		amqp.Publishing{
			ContentType: "text/csv",
			Body:        data,
			Headers:     amqp.Table{"task_id": taskID},
		})
	if err != nil {
		log.Printf("ERROR publish to RabbitMQ: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	log.Printf("Task %s published to queue", taskID)

	if err := rdb.SetEX(context.Background(), "task:"+taskID, `{"status":"pending"}`, time.Hour).Err(); err != nil {
		log.Printf("WARN redis set: %v", err)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"task_id": taskID})
}

func taskHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]
	log.Printf("Checking status for task %s", id)

	// 1. Redis
	val, err := rdb.Get(context.Background(), "task:"+id).Result()
	if err == nil {
		log.Printf("Task %s found in Redis", id)
		var data map[string]interface{}
		json.Unmarshal([]byte(val), &data)
		json.NewEncoder(w).Encode(data)
		return
	}
	log.Printf("Task %s not in Redis: %v", id, err)

	// 2. PostgreSQL
	var status, resultJSON, errorMsg string
	var createdAt time.Time
	row := db.QueryRow(`
		SELECT status, COALESCE(result::text, ''), COALESCE(error, ''), created_at
		FROM optimization_results
		WHERE task_id = $1
	`, id)
	err = row.Scan(&status, &resultJSON, &errorMsg, &createdAt)
	if err == nil {
		log.Printf("Task %s found in DB, status=%s", id, status)
		resp := map[string]interface{}{
			"id":         id,
			"status":     status,
			"created_at": createdAt,
		}
		if status == "completed" && resultJSON != "" {
			resp["result"] = json.RawMessage(resultJSON)
		} else if status == "failed" {
			resp["error"] = errorMsg
		}
		json.NewEncoder(w).Encode(resp)
		return
	}
	log.Printf("Task %s not in DB: %v", id, err)

	// 3. Memory
	if task, ok := tasks[id]; ok {
		log.Printf("Task %s found in memory", id)
		json.NewEncoder(w).Encode(task)
		return
	}

	log.Printf("Task %s not found anywhere", id)
	http.NotFound(w, r)
}
