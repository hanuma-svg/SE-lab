import { useEffect, useMemo, useState } from "react";
import "./styles.css";

type Task = { task_id: string; description: string; difficulty: string; category: string; adversarial: boolean; target_tests: string[]; retained_tests: string[]; allowed_write_paths: string[]; protected_paths: string[] };
type RunState = { run_id: string; experiment_id: string; task_id: string; variant: string; status: string; result?: Result; error?: string };
type RunRecord = { variant: string; status: string; pass_at_1: boolean; failure_class: string; model_calls: number; tool_calls: number; retries: number; wall_clock_duration_seconds?: number; budget_match: boolean; evidence_digest?: string };
type Result = { experiment_id: string; runs: RunRecord[]; mismatches: string[]; aggregation: { variants?: Record<string, { sample_count: number; pass_rate: number; pass_rate_ci95?: { low: number; high: number }; mean_wall_clock_duration_seconds?: number; mean_model_calls?: number; mean_tool_calls?: number; failure_classes?: Record<string, number> }> }; report: { task_id?: string; provider?: { name?: string; model?: string }; scientific_note?: string } };
type History = { experiment_id: string; task_id?: string; variants?: string[]; status: string; sample_count: number; created_at: string };

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? `Request failed (${response.status})`);
  return response.json();
};

function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [history, setHistory] = useState<History[]>([]);
  const [taskId, setTaskId] = useState("");
  const [variant, setVariant] = useState<"baseline" | "multi_agent" | "both">("both");
  const [run, setRun] = useState<RunState | null>(null);
  const [selectedResult, setSelectedResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const selectedTask = useMemo(() => tasks.find((task) => task.task_id === taskId), [tasks, taskId]);

  useEffect(() => {
    Promise.all([api<{ tasks: Task[] }>("/api/tasks"), api<{ experiments: History[] }>("/api/experiments")])
      .then(([taskPayload, historyPayload]) => {
        setTasks(taskPayload.tasks);
        setTaskId(taskPayload.tasks[0]?.task_id ?? "");
        setHistory(historyPayload.experiments);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!run || ["COMPLETED", "FAILED"].includes(run.status)) return;
    const timer = window.setInterval(() => {
      api<RunState>(`/api/runs/${run.run_id}`).then(setRun).catch((reason: Error) => setError(reason.message));
    }, 800);
    return () => window.clearInterval(timer);
  }, [run]);

  useEffect(() => {
    if (run?.status !== "COMPLETED") return;
    api<{ experiments: History[] }>("/api/experiments").then((payload) => setHistory(payload.experiments)).catch(() => undefined);
  }, [run?.status]);

  const startRun = async () => {
    setBusy(true);
    setError("");
    setSelectedResult(null);
    try {
      const next = await api<RunState>("/api/runs", { method: "POST", body: JSON.stringify({ task_id: taskId, variant }) });
      setRun(next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start run");
    } finally {
      setBusy(false);
    }
  };

  const openHistory = async (experimentId: string) => {
    try {
      setSelectedResult(await api<Result>(`/api/experiments/${experimentId}`));
      setRun(null);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load experiment");
    }
  };

  const result = run?.result ?? selectedResult;
  return (
    <div className="shell">
      <header className="topbar"><a className="wordmark" href="/">SE-Lab</a><span>Experiment console</span><a className="github" href="https://github.com/hanuma-svg/SE-lab" target="_blank" rel="noreferrer">GitHub ↗</a></header>
      <main>
        <section className="intro"><p className="kicker">Bounded multi-agent software engineering</p><h1>Evaluation platform</h1><p className="lede">Compare a deterministic baseline with a bounded multi-agent workflow on curated software-engineering tasks. Each run records execution evidence and is evaluated independently.</p><div className="architecture"><span>Benchmark</span><b>→</b><span>Baseline / Multi-Agent</span><b>→</b><span>Policy + Execution</span><b>→</b><span>Evidence + Evaluation</span><b>→</b><span>Results + Statistics</span></div></section>
        <section className="section console"><div className="section-title"><h2>Experiment console</h2><span className="label">mock provider / bounded run</span></div><div className="form-grid"><label>Benchmark task<select value={taskId} onChange={(event) => setTaskId(event.target.value)}>{tasks.map((task) => <option key={task.task_id} value={task.task_id}>{task.task_id}</option>)}</select></label><fieldset><legend>Variant</legend><label className="radio"><input type="radio" checked={variant === "baseline"} onChange={() => setVariant("baseline")} /> Baseline</label><label className="radio"><input type="radio" checked={variant === "multi_agent"} onChange={() => setVariant("multi_agent")} /> Multi-agent</label><label className="radio"><input type="radio" checked={variant === "both"} onChange={() => setVariant("both")} /> Both (comparison)</label></fieldset></div>{selectedTask && <p className="task-note"><span className="mono">{selectedTask.category}</span> {selectedTask.description} Write scope: <span className="mono">{selectedTask.allowed_write_paths.join(", ")}</span></p>}<button className="primary" disabled={!taskId || busy || (!!run && !["COMPLETED", "FAILED"].includes(run.status))} onClick={startRun}>{busy ? "Starting…" : "Run experiment"}</button></section>
        {error && <p className="error">{error}</p>}
        {run && <section className="section"><div className="section-title"><h2>Run status</h2><span className={`status ${run.status.toLowerCase()}`}>● {run.status}</span></div><dl className="definition"><dt>Run ID</dt><dd>{run.run_id}</dd><dt>Task</dt><dd>{run.task_id}</dd><dt>Variant</dt><dd>{run.variant}</dd></dl>{run.error && <p className="error">{run.error}</p>}</section>}
        {result && <ResultView result={result} />}
        <section className="section"><div className="section-title"><h2>Experiment history</h2><span className="label">SQLite registry</span></div><table><thead><tr><th>Experiment ID</th><th>Task</th><th>Variant</th><th>Status</th><th>Created</th></tr></thead><tbody>{history.length ? history.map((item) => <tr key={item.experiment_id}><td><button className="link mono" onClick={() => openHistory(item.experiment_id)}>{item.experiment_id}</button></td><td className="mono">{item.task_id ?? "—"}</td><td>{item.variants?.join(", ") ?? "—"}</td><td><span className="status pass">● {item.status}</span></td><td>{item.created_at}</td></tr>) : <tr><td colSpan={5} className="muted">No persisted web experiments yet.</td></tr>}</tbody></table></section>
        <section className="section scope"><h2>Scope</h2><p>Current public demo uses:</p><ul><li>deterministic/mock provider</li><li>curated smoke benchmark</li><li>bounded execution</li><li>descriptive statistics</li></ul><p className="note">This demo demonstrates the evaluation infrastructure; it does not establish multi-agent superiority.</p></section>
      </main>
      <footer>Python · FastAPI · React/TypeScript · SQLite · policy gateway · independent evidence evaluation · automated tests</footer>
    </div>
  );
}

function ResultView({ result }: { result: Result }) {
  const variants = Object.entries(result.aggregation.variants ?? {});
  return <section className="section"><div className="section-title"><h2>Results</h2><span className="label mono">{result.experiment_id}</span></div><p className="muted">Task <span className="mono">{result.report.task_id}</span> · provider <span className="mono">{result.report.provider?.name}</span></p><table className="comparison"><thead><tr><th>Metric</th>{variants.map(([name]) => <th key={name}>{name}</th>)}</tr></thead><tbody><MetricRow label="Pass@1" values={variants.map(([, values]) => `${values.sample_count * values.pass_rate} / ${values.sample_count}`)} /><MetricRow label="95% Wilson CI" values={variants.map(([, values]) => values.pass_rate_ci95 ? `[${values.pass_rate_ci95.low.toFixed(3)}, ${values.pass_rate_ci95.high.toFixed(3)}]` : "—")} /><MetricRow label="Mean wall time" values={variants.map(([, values]) => `${values.mean_wall_clock_duration_seconds?.toFixed(3) ?? "—"} s`)} /><MetricRow label="Model calls" values={variants.map(([, values]) => values.mean_model_calls?.toFixed(1) ?? "—")} /><MetricRow label="Tool calls" values={variants.map(([, values]) => values.mean_tool_calls?.toFixed(1) ?? "—")} /><MetricRow label="Failure class" values={variants.map(([, values]) => Object.entries(values.failure_classes ?? {}).map(([key, count]) => `${key}: ${count}`).join(", ") || "none recorded")} /></tbody></table><p className="note">{result.report.scientific_note ?? "Descriptive result from the current bounded benchmark."}</p></section>;
}

function MetricRow({ label, values }: { label: string; values: string[] }) { return <tr><td>{label}</td>{values.map((value, index) => <td key={`${label}-${index}`} className="mono">{value}</td>)}</tr>; }

export default App;
