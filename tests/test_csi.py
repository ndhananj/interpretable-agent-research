from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import csi
from interpretability.resources import ResourceSnapshot


@dataclass
class FakeScore:
    functionality: float
    explainability: float
    accepted: bool
    reason: str
    details: dict


def test_state_and_events_helpers(tmp_path: Path) -> None:
    csi.update_state(tmp_path, {"status": "running", "total_trials": 1})
    csi.update_state(tmp_path, {"accepted_trials": 1})
    csi.append_event(tmp_path, "trial_scored", {"trial": 1})

    assert csi.read_state(tmp_path)["status"] == "running"
    assert csi.read_state(tmp_path)["accepted_trials"] == 1
    events = csi.read_events(tmp_path)
    assert events[0]["event"] == "trial_scored"
    assert events[0]["trial"] == 1


def test_stale_pid_cleanup_and_stop_request(tmp_path: Path, monkeypatch) -> None:
    csi.write_pid(tmp_path, 999999)
    monkeypatch.setattr(csi, "process_alive", lambda pid: False)

    assert csi.cleanup_stale_pid(tmp_path)
    assert csi.read_pid(tmp_path) is None

    csi.request_stop(tmp_path)
    assert csi.stop_requested(tmp_path)


def test_daemon_iteration_backoff_writes_state_and_event(tmp_path: Path, monkeypatch) -> None:
    config = {
        "resources": {"max_load_ratio": 0.5, "min_available_memory_gib": 4},
        "scoring": {"task_paths": ["tasks/replace_token.yaml"]},
    }
    snapshot = ResourceSnapshot(4, 8.0, 2.0, 16.0, 12.0, False, 0, False, False)
    monkeypatch.setattr(csi, "detect_resources", lambda: snapshot)

    state: dict = {}
    did_work = csi.daemon_iteration(config, tmp_path / "csi", tmp_path, state)

    assert not did_work
    written = csi.read_state(tmp_path / "csi")
    assert written["status"] == "backoff"
    assert "load ratio" in written["backoff_reason"]
    assert csi.read_events(tmp_path / "csi")[0]["event"] == "backoff"


def test_daemon_iteration_runs_one_trial_cycle(tmp_path: Path, monkeypatch) -> None:
    config = {
        "resources": {"max_load_ratio": 0.75, "min_available_memory_gib": 4},
        "model": {"backend": "mock"},
        "experiment": {"functionality_floor_ratio": 0.9},
        "scoring": {"task_paths": ["tasks/replace_token.yaml"], "metrics_config": "configs/metrics.yaml"},
    }
    snapshot = ResourceSnapshot(4, 1.0, 0.25, 16.0, 12.0, False, 0, False, False)
    monkeypatch.setattr(csi, "detect_resources", lambda: snapshot)
    monkeypatch.setattr(csi, "make_backend", lambda cfg: object())
    monkeypatch.setattr(csi, "load_yaml", lambda path: {})

    def baseline(*args, **kwargs):
        run_dir = tmp_path / "baseline-run"
        run_dir.mkdir()
        return FakeScore(1.0, 0.0, True, "baseline", {}), run_dir

    def trial(*args, **kwargs):
        run_dir = tmp_path / "trial-run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(json.dumps({"functionality": 1.0}), encoding="utf-8")
        (run_dir / "score.json").write_text(
            json.dumps({"explainability": 0.8, "accepted": True, "reason": "accepted"}),
            encoding="utf-8",
        )
        return FakeScore(1.0, 0.8, True, "accepted", {}), run_dir

    monkeypatch.setattr(csi, "run_baseline_cycle", baseline)
    monkeypatch.setattr(csi, "run_one_trial_cycle", trial)

    state: dict = {}
    did_work = csi.daemon_iteration(config, tmp_path / "csi", tmp_path, state)

    assert did_work
    written = csi.read_state(tmp_path / "csi")
    assert written["baseline_functionality"] == 1.0
    assert written["total_trials"] == 1
    assert written["accepted_trials"] == 1
    assert (tmp_path / "summary.json").exists()


def test_dashboard_renders_fixture_state(tmp_path: Path) -> None:
    csi_dir = tmp_path / "csi"
    run_dir = tmp_path / "20260101T000000-trial-1-123"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({"functionality": 1.0}), encoding="utf-8")
    (run_dir / "score.json").write_text(
        json.dumps({"explainability": 0.7, "accepted": True, "reason": "improved"}),
        encoding="utf-8",
    )
    csi.update_state(csi_dir, {"status": "running", "pid": 123, "total_trials": 1, "accepted_trials": 1})
    csi.append_event(csi_dir, "accepted", {"trial": 1})

    html = csi.render_dashboard(csi_dir, tmp_path)

    assert "Continuous Self-Improvement" in html
    assert "running" in html
    assert "20260101T000000-trial-1-123" in html
    assert "accepted" in html
