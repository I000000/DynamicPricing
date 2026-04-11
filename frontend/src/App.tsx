import { useState } from 'react';
import './App.css';

interface OptimizationResult {
  product_key: number;
  current_price: number;
  unit_cost: number;
  current_profit: number;
  optimal_price: number;
  expected_demand: number;
  expected_profit: number;
}

interface TaskResponse {
  id: string;
  status: string;
  created_at: string;
  result?: OptimizationResult[];
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [results, setResults] = useState<OptimizationResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Выберите файл');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    setStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Ошибка загрузки');
      const data = await response.json();
      setTaskId(data.task_id);
      setStatus('pending');
      pollTask(data.task_id);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  const pollTask = async (id: string) => {
    const maxAttempts = 30;
    let attempts = 0;

    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/task/${id}`);
        if (!response.ok) throw new Error('Ошибка получения статуса');
        const data: TaskResponse = await response.json();
        setStatus(data.status);

        if (data.status === 'completed') {
          setResults(data.result || []);
          setLoading(false);
        } else if (data.status === 'failed') {
          setError('Обработка завершилась с ошибкой');
          setLoading(false);
        } else {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(checkStatus, 2000);
          } else {
            setError('Превышено время ожидания');
            setLoading(false);
          }
        }
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };

    checkStatus();
  };

  return (
    <div className="container">
      <h1>Динамическое ценообразование</h1>
      <div className="upload-section">
        <input type="file" accept=".csv" onChange={handleFileChange} disabled={loading} />
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Обработка...' : 'Загрузить и оптимизировать'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {taskId && (
        <div className="task-info">
          <strong>Task ID:</strong> {taskId} <br />
          <strong>Статус:</strong> {status}
        </div>
      )}

      {results && (
        <div className="results">
          <h2>Результаты оптимизации</h2>
          <table>
            <thead>
              <tr>
                <th>Товар</th>
                <th>Текущая цена</th>
                <th>Себестоимость</th>
                <th>Оптимальная цена</th>
                <th>Ожидаемый спрос</th>
                <th>Текущая прибыль</th>
                <th>Ожидаемая прибыль</th>
                <th>Прирост</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.product_key}>
                  <td>{r.product_key}</td>
                  <td>{r.current_price.toFixed(2)}</td>
                  <td>{r.unit_cost.toFixed(2)}</td>
                  <td>{r.optimal_price.toFixed(2)}</td>
                  <td>{r.expected_demand.toFixed(2)}</td>
                  <td>{r.current_profit.toFixed(2)}</td>
                  <td>{r.expected_profit.toFixed(2)}</td>
                  <td style={{ color: r.expected_profit >= r.current_profit ? '#15803d' : '#b91c1c', fontWeight: 500 }}>
                    {r.current_profit > 0 
                      ? `${(((r.expected_profit - r.current_profit) / r.current_profit) * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;