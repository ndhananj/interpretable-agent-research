from __future__ import annotations

import argparse
import html
import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agent_harness.backends import make_backend
from interpretability.config import load_yaml
from interpretability.resources import configure_conservative_threads, detect_resources, should_backoff
from run_experiment import run_baseline_cycle, run_one_trial_cycle


CSI_DIRNAME = "csi"
PID_FILE = "daemon.pid"
STOP_FILE = "stop.requested"
STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous self-improvement local control")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sleep-s", type=float, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "stop", "restart", "status"):
        subparsers.add_parser(command)
    dashboard = subparsers.add_parser("dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    daemon = subparsers.add_parser("_daemon")
    daemon.add_argument("--foreground", action="store_true")
    args = parser.parse_args()

    if args.command == "start":
        start_daemon(args.config, args.sleep_s)
    elif args.command == "stop":
        stop_daemon(args.config)
    elif args.command == "restart":
        stop_daemon(args.config)
        start_daemon(args.config, args.sleep_s)
    elif args.command == "status":
        print_status(args.config)
    elif args.command == "dashboard":
        serve_dashboard(args.config, args.host, args.port)
    elif args.command == "_daemon":
        run_daemon(args.config, args.sleep_s)


def start_daemon(config_path: str, sleep_s: float | None) -> None:
    csi_dir = get_csi_dir(config_path)
    csi_dir.mkdir(parents=True, exist_ok=True)
    pid = read_pid(csi_dir)
    if pid and process_alive(pid):
        print(f"CSI daemon already running with PID {pid}")
        return
    cleanup_stale_pid(csi_dir)
    stop_path(csi_dir).unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", config_path]
    if sleep_s is not None:
        cmd.extend(["--sleep-s", str(sleep_s)])
    cmd.append("_daemon")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    write_pid(csi_dir, proc.pid)
    update_state(csi_dir, {"status": "starting", "pid": proc.pid, "updated_at": now()})
    append_event(csi_dir, "daemon_started", {"pid": proc.pid})
    print(f"CSI daemon started with PID {proc.pid}")


def stop_daemon(config_path: str, timeout_s: float = 10.0) -> None:
    csi_dir = get_csi_dir(config_path)
    csi_dir.mkdir(parents=True, exist_ok=True)
    request_stop(csi_dir)
    pid = read_pid(csi_dir)
    if pid and process_alive(pid):
        deadline = time.time() + timeout_s
        while time.time() < deadline and process_alive(pid):
            time.sleep(0.2)
        if process_alive(pid):
            os.kill(pid, signal.SIGTERM)
    cleanup_stale_pid(csi_dir)
    update_state(csi_dir, {"status": "stopped", "pid": None, "updated_at": now()})
    append_event(csi_dir, "stop_requested", {})
    print("CSI daemon stopped")


def print_status(config_path: str) -> None:
    csi_dir = get_csi_dir(config_path)
    state = read_state(csi_dir)
    pid = read_pid(csi_dir)
    alive = bool(pid and process_alive(pid))
    print(f"status: {state.get('status', 'unknown')}")
    print(f"pid: {pid or '-'} ({'alive' if alive else 'not running'})")
    print(f"trials: {state.get('total_trials', 0)}")
    print(f"accepted: {state.get('accepted_trials', 0)}")
    print(f"latest_run: {state.get('latest_run', '-')}")
    print(f"last_error: {state.get('last_error', '-')}")
    print(f"backoff_reason: {state.get('backoff_reason', '-')}")


def run_daemon(config_path: str, sleep_s: float | None = None, once: bool = False) -> None:
    config = load_yaml(config_path)
    experiment_cfg = config.get("experiment", {})
    resources_cfg = config.get("resources", {})
    configure_conservative_threads(int(resources_cfg.get("max_cpu_threads", 4)))
    run_root = Path(experiment_cfg.get("run_root", "runs"))
    run_root.mkdir(parents=True, exist_ok=True)
    csi_dir = run_root / CSI_DIRNAME
    csi_dir.mkdir(parents=True, exist_ok=True)
    write_pid(csi_dir, os.getpid())
    sleep_s = float(sleep_s if sleep_s is not None else experiment_cfg.get("daemon_sleep_s", 30))
    state = read_state(csi_dir)
    state.update({"status": "running", "pid": os.getpid(), "config": config_path, "updated_at": now()})
    update_state(csi_dir, state)

    try:
        while not stop_requested(csi_dir):
            did_work = daemon_iteration(config, csi_dir, run_root, state)
            if once:
                break
            if not did_work:
                time.sleep(sleep_s)
        state.update({"status": "stopped", "pid": None, "updated_at": now()})
        update_state(csi_dir, state)
        append_event(csi_dir, "stopped", {})
    except Exception as exc:
        state.update({"status": "error", "last_error": str(exc), "updated_at": now()})
        update_state(csi_dir, state)
        append_event(csi_dir, "error", {"message": str(exc)})
        raise
    finally:
        if read_pid(csi_dir) == os.getpid():
            pid_path(csi_dir).unlink(missing_ok=True)


def daemon_iteration(
    config: dict[str, Any],
    csi_dir: Path,
    run_root: Path,
    state: dict[str, Any],
    backend: Any | None = None,
    metrics_config: dict[str, Any] | None = None,
) -> bool:
    resources_cfg = config.get("resources", {})
    snapshot = detect_resources()
    backoff, reason = should_backoff(resources_cfg, snapshot)
    state["resource_snapshot"] = snapshot.__dict__
    if backoff:
        state.update({"status": "backoff", "backoff_reason": reason, "updated_at": now()})
        update_state(csi_dir, state)
        append_event(csi_dir, "backoff", {"reason": reason, "resources": snapshot.__dict__})
        return False

    task_paths = config.get("scoring", {}).get("task_paths", [])
    if not task_paths:
        raise SystemExit("No task_paths configured")
    backend = backend or make_backend(config.get("model", {}))
    metrics_config = metrics_config or load_yaml(config.get("scoring", {}).get("metrics_config", "configs/metrics.yaml"))
    floor_ratio = float(config.get("experiment", {}).get("functionality_floor_ratio", 0.90))
    baseline = state.get("baseline_functionality") or config.get("experiment", {}).get("baseline_functionality")
    if baseline is None:
        append_event(csi_dir, "baseline_started", {})
        result, run_dir = run_baseline_cycle(run_root, backend, task_paths, config, metrics_config)
        baseline = result.functionality
        state.update({"baseline_functionality": baseline, "latest_run": str(run_dir), "updated_at": now()})
        append_event(csi_dir, "baseline_scored", score_payload(result, run_dir))

    trial_number = int(state.get("total_trials", 0)) + 1
    incumbent = float(state.get("incumbent_explainability", 0.0))
    append_event(csi_dir, "trial_started", {"trial": trial_number})
    result, run_dir = run_one_trial_cycle(run_root, backend, task_paths, config, metrics_config, baseline, incumbent, trial_number, floor_ratio)
    append_event(csi_dir, "trial_scored", score_payload(result, run_dir, trial_number))
    accepted = list(state.get("accepted", []))
    if result.accepted:
        accepted.append(result.__dict__)
        state["incumbent_explainability"] = result.explainability
        append_event(csi_dir, "accepted", score_payload(result, run_dir, trial_number))
    else:
        append_event(csi_dir, "rejected", score_payload(result, run_dir, trial_number))
    state.update(
        {
            "status": "running",
            "backoff_reason": None,
            "last_error": None,
            "total_trials": trial_number,
            "accepted_trials": len(accepted),
            "accepted": accepted,
            "latest_run": str(run_dir),
            "updated_at": now(),
        }
    )
    update_state(csi_dir, state)
    write_summary(run_root, baseline, accepted, snapshot.__dict__)
    return True


def serve_dashboard(config_path: str, host: str, port: int) -> None:
    csi_dir = get_csi_dir(config_path)
    run_root = csi_dir.parent

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = render_dashboard(csi_dir, run_root).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving CSI dashboard at http://{host}:{port}")
    server.serve_forever()


def render_dashboard(csi_dir: Path, run_root: Path) -> str:
    state = read_state(csi_dir)
    events = read_events(csi_dir, limit=50)
    rows = trend_rows(run_root)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CSI Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 32px; color: #202124; background: #f7f8fa; }}
main {{ max-width: 1100px; margin: 0 auto; }}
section {{ background: white; border: 1px solid #d7dce2; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
h1, h2 {{ margin: 0 0 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.metric {{ border: 1px solid #e1e5ea; border-radius: 6px; padding: 12px; }}
.value {{ font-size: 24px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; border-bottom: 1px solid #e1e5ea; padding: 8px; font-size: 14px; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body><main>
<h1>Continuous Self-Improvement</h1>
<section class="grid">
{metric("Status", state.get("status", "unknown"))}
{metric("PID", state.get("pid", "-"))}
{metric("Trials", state.get("total_trials", 0))}
{metric("Accepted", state.get("accepted_trials", 0))}
</section>
<section><h2>Current Activity</h2>
<p>Latest run: <code>{esc(state.get("latest_run", "-"))}</code></p>
<p>Backoff: {esc(state.get("backoff_reason") or "-")}</p>
<p>Last error: {esc(state.get("last_error") or "-")}</p>
</section>
<section><h2>Trend</h2><table><tr><th>Run</th><th>Functionality</th><th>Explainability</th><th>Accepted</th><th>Reason</th></tr>{''.join(rows)}</table></section>
<section><h2>Recent Events</h2><table><tr><th>Time</th><th>Event</th><th>Details</th></tr>{''.join(event_row(e) for e in events)}</table></section>
</main></body></html>"""


def trend_rows(run_root: Path) -> list[str]:
    rows = []
    for run_dir in sorted([p for p in run_root.iterdir() if p.is_dir() and p.name != CSI_DIRNAME], reverse=True)[:20]:
        metrics = read_json(run_dir / "metrics.json")
        score = read_json(run_dir / "score.json")
        rows.append(
            "<tr>"
            f"<td><code>{esc(run_dir.name)}</code></td>"
            f"<td>{esc(metrics.get('functionality', '-'))}</td>"
            f"<td>{esc(score.get('explainability', '-'))}</td>"
            f"<td>{esc(score.get('accepted', '-'))}</td>"
            f"<td>{esc(score.get('reason', '-'))}</td>"
            "</tr>"
        )
    return rows


def event_row(event: dict[str, Any]) -> str:
    details = {k: v for k, v in event.items() if k not in {"time", "event"}}
    return f"<tr><td>{esc(event.get('time', '-'))}</td><td>{esc(event.get('event', '-'))}</td><td><code>{esc(json.dumps(details, sort_keys=True))}</code></td></tr>"


def metric(label: str, value: Any) -> str:
    return f'<div class="metric"><div>{esc(label)}</div><div class="value">{esc(value)}</div></div>'


def get_csi_dir(config_path: str) -> Path:
    config = load_yaml(config_path)
    return Path(config.get("experiment", {}).get("run_root", "runs")) / CSI_DIRNAME


def pid_path(csi_dir: Path) -> Path:
    return csi_dir / PID_FILE


def stop_path(csi_dir: Path) -> Path:
    return csi_dir / STOP_FILE


def read_pid(csi_dir: Path) -> int | None:
    try:
        return int(pid_path(csi_dir).read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_pid(csi_dir: Path, pid: int) -> None:
    pid_path(csi_dir).write_text(f"{pid}\n", encoding="utf-8")


def cleanup_stale_pid(csi_dir: Path) -> bool:
    pid = read_pid(csi_dir)
    if pid and process_alive(pid):
        return False
    pid_path(csi_dir).unlink(missing_ok=True)
    return True


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def request_stop(csi_dir: Path) -> None:
    stop_path(csi_dir).write_text(now() + "\n", encoding="utf-8")


def stop_requested(csi_dir: Path) -> bool:
    return stop_path(csi_dir).exists()


def read_state(csi_dir: Path) -> dict[str, Any]:
    return read_json(csi_dir / STATE_FILE)


def update_state(csi_dir: Path, updates: dict[str, Any]) -> None:
    csi_dir.mkdir(parents=True, exist_ok=True)
    state = read_state(csi_dir)
    state.update(updates)
    (csi_dir / STATE_FILE).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(csi_dir: Path, event: str, payload: dict[str, Any]) -> None:
    csi_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": now(), "event": event, **payload}
    with (csi_dir / EVENTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_events(csi_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    path = csi_dir / EVENTS_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(events))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_summary(run_root: Path, baseline: float, accepted: list[dict[str, Any]], resources: dict[str, Any]) -> None:
    summary = {"baseline_functionality": baseline, "accepted": accepted, "resources": resources}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_payload(result: Any, run_dir: Path, trial: int | None = None) -> dict[str, Any]:
    payload = {"run_dir": str(run_dir), **result.__dict__}
    if trial is not None:
        payload["trial"] = trial
    return payload


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def esc(value: Any) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()
