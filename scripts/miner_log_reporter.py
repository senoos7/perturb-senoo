#!/usr/bin/env python3
"""
Tail Perturb miner PM2 logs and post high-signal summaries to Slack.

Usage:
  cp scripts/miner_slack.env.example scripts/miner_slack.env
  # set SLACK_WEBHOOK_URL
  python scripts/miner_log_reporter.py

Or via PM2:
  bash scripts/run_miner_slack_watch.sh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]

RE_FORWARD = re.compile(
    r"\[STEP_START\] miner_forward task_id=(?P<task_id>\S+) norm_type=\S+ epsilon=(?P<epsilon>[\d.]+)"
)
RE_ATTACK_START = re.compile(
    r"\[SMA-MI-FGSM\] start .*"
    r"image_shape=\((?P<c>\d+),\s*(?P<h>\d+),\s*(?P<w>\d+)\).*"
    r"uses_work=(?P<uses_work>\w+)"
)
RE_FINISHED = re.compile(
    r"Finished task=(?P<task_id>\S+) .*"
    r"target_idx=(?P<target_idx>\d+) final_pred=(?P<final_pred>\d+) "
    r"best_delta=(?P<linf>[\d.]+) best_rmse=(?P<rmse>[\d.]+)"
)
RE_DONE = re.compile(
    r"\[SMA-MI-FGSM\] done .*"
    r"elapsed_ms=(?P<elapsed_ms>\d+) .*"
    r"stop=(?P<stop>\S+) linf=(?P<linf>[\d.]+) rmse=(?P<rmse>[\d.]+) "
    r"pred_idx=(?P<pred_idx>\d+) target_idx=(?P<target_idx>\d+) "
    r"misclassified=(?P<misclassified>\w+)"
)
RE_ERROR = re.compile(r"(Traceback \(most recent|RuntimeError|ERROR \|)", re.I)


@dataclass
class TaskContext:
    task_id: str
    epsilon: float = 0.0
    shape: str = ""
    uses_work: bool = False
    full_res_miss: bool = False
    stop: str = ""
    elapsed_ms: int = 0
    target_idx: int | None = None
    final_pred: int | None = None


@dataclass
class ReporterState:
    pending: dict[str, TaskContext] = field(default_factory=dict)
    task_order: list[str] = field(default_factory=list)
    last_event_ts: float = field(default_factory=time.time)
    last_slack_ts: float = 0.0
    last_heartbeat_ts: float = 0.0


def _latest_task(state: ReporterState) -> TaskContext | None:
    while state.task_order:
        task_id = state.task_order[-1]
        ctx = state.pending.get(task_id)
        if ctx is not None:
            return ctx
        state.task_order.pop()
    return None


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        os.environ.setdefault(key, value)


def _classify_outcome(*, linf: float, rmse: float, misclassified: bool, elapsed_ms: int) -> tuple[str, str]:
    if not misclassified or linf <= 0.0:
        return "FAIL", "no_effect"
    if elapsed_ms >= 11_000:
        return "WARN", "timeout_risk"
    if linf <= 0.0045 and rmse <= 0.001:
        return "OK", "good_score"
    if linf <= 0.0045 and rmse <= 0.002:
        return "OK", "acceptable"
    return "LOW", "high_linf_or_rmse"


def _emoji(level: str) -> str:
    return {"OK": ":white_check_mark:", "LOW": ":warning:", "WARN": ":hourglass:", "FAIL": ":x:"}.get(level, ":grey_question:")


def _post_slack(webhook_url: str, text: str, min_interval_s: float, state: ReporterState) -> None:
    now = time.time()
    if now - state.last_slack_ts < min_interval_s:
        return
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status >= 300:
                print(f"[slack] unexpected status={response.status}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"[slack] post failed: {exc}", file=sys.stderr)
        return
    state.last_slack_ts = now


def _resolve_log_paths(process_name: str) -> list[Path]:
    pm2_home = Path(os.getenv("PM2_HOME", Path.home() / ".pm2"))
    logs_dir = pm2_home / "logs"
    candidates = [
        logs_dir / f"{process_name}-out.log",
        logs_dir / f"{process_name}-error.log",
        logs_dir / f"{process_name}.log",
    ]
    return [path for path in candidates if path.is_file()]


def _follow_files(paths: list[Path]) -> TextIO:
    import subprocess

    if not paths:
        raise FileNotFoundError(
            "No PM2 log files found. Start the miner first: bash ./scripts/run_miner.sh"
        )
    cmd = ["tail", "-F", "-n", "0", *[str(p) for p in paths]]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ).stdout  # type: ignore[return-value]


def _format_task_message(ctx: TaskContext, *, linf: float, rmse: float, misclassified: bool) -> str:
    level, reason = _classify_outcome(
        linf=linf,
        rmse=rmse,
        misclassified=misclassified,
        elapsed_ms=ctx.elapsed_ms,
    )
    lines = [
        f"{_emoji(level)} *Perturb miner task* `{ctx.task_id[:24]}...`" if len(ctx.task_id) > 24 else f"{_emoji(level)} *Perturb miner task* `{ctx.task_id}`",
        f"• outcome: *{level}* ({reason})",
        f"• image: `{ctx.shape or 'unknown'}` uses_work={ctx.uses_work}",
        f"• epsilon: `{ctx.epsilon}`",
        f"• attack_ms: `{ctx.elapsed_ms}` stop: `{ctx.stop or '-'}`",
        f"• linf: `{linf:.6f}` rmse: `{rmse:.6f}`",
        f"• label idx: `{ctx.target_idx if ctx.target_idx is not None else '?'}` → `{ctx.final_pred if ctx.final_pred is not None else '?'}`",
    ]
    if ctx.full_res_miss:
        lines.append("• :warning: *full_res_miss_after_project* (upsample failed)")
    if level == "WARN":
        lines.append("• :warning: near validator timeout (~15s) — 408 risk")
    if level == "FAIL" and linf <= 0.0:
        lines.append("• validator score likely *0* (no perturbation)")
    elif level == "LOW":
        lines.append("• misclassified but *low score* expected (linf/rmse too high)")
    elif level == "OK":
        lines.append("• competitive range if gates pass")
    return "\n".join(lines)


def _handle_line(line: str, args: argparse.Namespace, state: ReporterState) -> None:
    state.last_event_ts = time.time()
    clean = line.strip()
    if not clean:
        return

    if RE_ERROR.search(clean):
        snippet = clean[-500:] if len(clean) > 500 else clean
        _post_slack(
            args.webhook_url,
            f":rotating_light: *Perturb miner error*\n```{snippet}```",
            args.min_post_interval_s,
            state,
        )
        return

    match = RE_FORWARD.search(clean)
    if match:
        task_id = match.group("task_id")
        state.pending[task_id] = TaskContext(task_id=task_id, epsilon=float(match.group("epsilon")))
        state.task_order.append(task_id)
        return

    match = RE_ATTACK_START.search(clean)
    if match:
        ctx = _latest_task(state)
        if ctx is not None and not ctx.shape:
            ctx.shape = f"{match.group('h')}x{match.group('w')}"
            ctx.uses_work = match.group("uses_work") == "True"
        return

    if "full_res_miss_after_project" in clean:
        ctx = _latest_task(state)
        if ctx is not None:
            ctx.full_res_miss = True
        return

    match = RE_DONE.search(clean)
    if match:
        ctx = _latest_task(state)
        if ctx is not None:
            ctx.stop = match.group("stop")
            ctx.elapsed_ms = int(match.group("elapsed_ms"))
            ctx.final_pred = int(match.group("pred_idx"))
            ctx.target_idx = int(match.group("target_idx"))
        return

    match = RE_FINISHED.search(clean)
    if match:
        task_id = match.group("task_id")
        ctx = state.pending.pop(task_id, TaskContext(task_id=task_id))
        if task_id in state.task_order:
            state.task_order.remove(task_id)
        ctx.final_pred = int(match.group("final_pred"))
        ctx.target_idx = int(match.group("target_idx"))
        linf = float(match.group("linf"))
        rmse = float(match.group("rmse"))
        misclassified = ctx.final_pred != ctx.target_idx
        msg = _format_task_message(ctx, linf=linf, rmse=rmse, misclassified=misclassified)
        _post_slack(args.webhook_url, msg, args.min_post_interval_s, state)
        return


def _maybe_heartbeat(args: argparse.Namespace, state: ReporterState) -> None:
    if args.heartbeat_seconds <= 0:
        return
    now = time.time()
    if now - state.last_heartbeat_ts < args.heartbeat_seconds:
        return
    idle_s = int(now - state.last_event_ts)
    _post_slack(
        args.webhook_url,
        f":satellite: *Perturb miner watcher alive*\n"
        f"• process: `{args.pm2_process}`\n"
        f"• last log event: `{idle_s}s` ago",
        args.min_post_interval_s,
        state,
    )
    state.last_heartbeat_ts = now


def main() -> int:
    parser = argparse.ArgumentParser(description="Post Perturb miner log summaries to Slack")
    parser.add_argument(
        "--env-file",
        default=os.getenv("MINER_SLACK_ENV", str(ROOT / "scripts" / "miner_slack.env")),
        help="Env file with SLACK_WEBHOOK_URL",
    )
    parser.add_argument(
        "--pm2-process",
        default=os.getenv("PM2_MINER_PROCESS", "perturb-miner"),
        help="PM2 process name to tail logs from",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=int(os.getenv("SLACK_HEARTBEAT_SECONDS", "1800")),
        help="Post alive ping every N seconds (0=off)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("SLACK_WEBHOOK_URL is not set. Copy scripts/miner_slack.env.example to scripts/miner_slack.env", file=sys.stderr)
        return 1

    args.webhook_url = webhook_url
    args.min_post_interval_s = float(os.getenv("SLACK_MIN_POST_INTERVAL_SECONDS", "2"))

    log_paths = _resolve_log_paths(args.pm2_process)
    print(f"[miner_log_reporter] tailing: {', '.join(str(p) for p in log_paths)}", flush=True)

    _post_slack(
        webhook_url,
        f":rocket: *Perturb Slack watcher started*\n• tailing `{args.pm2_process}` logs",
        0.0,
        ReporterState(),
    )

    state = ReporterState()
    stream = _follow_files(log_paths)
    try:
        while True:
            line = stream.readline()
            if line:
                _handle_line(line, args, state)
            else:
                time.sleep(0.5)
            _maybe_heartbeat(args, state)
    except KeyboardInterrupt:
        print("\n[miner_log_reporter] stopped", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
