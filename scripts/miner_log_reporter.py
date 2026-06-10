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

# Defaults mirror perturbnet/constants.py (avoid importing bittensor stack here).
DEFAULT_MIN_LINF_DELTA = 0.003
DEFAULT_MAX_LINF_DELTA = 0.03
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_LINF_WEIGHT = 0.7
DEFAULT_RMSE_WEIGHT = 0.3

RE_FORWARD = re.compile(
    r"\[STEP_START\] miner_forward task_id=(?P<task_id>\S+) norm_type=(?P<norm_type>\S+) epsilon=(?P<epsilon>[\d.]+)"
)
RE_ATTACK_START = re.compile(
    r"\[SMA-MI-FGSM\] start .*"
    r"image_shape=\((?P<c>\d+),\s*(?P<h>\d+),\s*(?P<w>\d+)\).*"
    r"work_shape=\((?P<wc>\d+),\s*(?P<wh>\d+),\s*(?P<ww>\d+)\).*"
    r"uses_work=(?P<uses_work>\w+)"
    r"(?: .*full_pixels=(?P<full_pixels>\d+))?"
    r".*target_idx=(?P<target_idx>\d+)"
    r".*min_delta=(?P<min_delta>[\d.]+)"
    r".*steps=(?P<steps>\d+) restarts=(?P<restarts>\d+)"
    r".*budget_ms=(?P<budget_ms>\d+|unlimited)"
)
RE_FINISHED = re.compile(
    r"Finished task=(?P<task_id>\S+) "
    r"algorithm=(?P<algorithm>\S+) key=(?P<algorithm_key>\S+) "
    r"target_idx=(?P<target_idx>\d+) final_pred=(?P<final_pred>\d+) "
    r"best_delta=(?P<linf>[\d.]+) best_rmse=(?P<rmse>[\d.]+) min_delta=(?P<min_delta>[\d.]+)"
)
RE_DONE = re.compile(
    r"\[SMA-MI-FGSM\] done .*"
    r"elapsed_ms=(?P<elapsed_ms>\d+)(?: remaining_ms=(?P<remaining_ms>-?\d+))?"
    r".*stop=(?P<stop>\S+) linf=(?P<linf>[\d.]+) rmse=(?P<rmse>[\d.]+) "
    r"pred_idx=(?P<pred_idx>\d+) target_idx=(?P<target_idx>\d+) "
    r"misclassified=(?P<misclassified>\w+)"
)
RE_ERROR = re.compile(r"(Traceback \(most recent|RuntimeError|ERROR \|)", re.I)
RE_BLACKLIST = re.compile(r"Blacklist reject: (?P<reason>.+)")
RE_SKIP = re.compile(r"Skipping task=(?P<task_id>\S+): (?P<reason>.+)")
RE_KV = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class TaskContext:
    task_id: str
    norm_type: str = "Linf"
    epsilon: float = 0.0
    prompt: str = ""
    true_label: str = ""
    shape: str = ""
    work_shape: str = ""
    uses_work: bool = False
    full_pixels: int = 0
    full_res_miss: bool = False
    stop: str = ""
    elapsed_ms: int = 0
    remaining_ms: int | None = None
    target_idx: int | None = None
    final_pred: int | None = None
    min_delta: float = DEFAULT_MIN_LINF_DELTA
    budget_ms: int | None = None
    steps: int = 0
    restarts_cfg: int = 0
    algorithm: str = ""
    algorithm_key: str = ""
    saliency_ms: int | None = None
    saliency_pixels: int | None = None
    sparse_result: str = ""
    sparse_ms: int | None = None
    sparse_topk: int | None = None
    sparse_tried: int | None = None
    restarts_used: int = 0
    best_update_sources: list[str] = field(default_factory=list)
    early_stop_stage: str = ""
    project_full_success: bool | None = None
    project_full_linf: float | None = None
    project_full_rmse: float | None = None
    forward_ts: float = 0.0
    misclassified_attack: bool | None = None
    done_linf: float | None = None
    done_rmse: float | None = None
    _saliency_start_ms: int | None = field(default=None, repr=False)
    _sparse_start_ms: int | None = field(default=None, repr=False)


@dataclass
class TaskRecord:
    ts: float
    level: str
    est_score: float
    linf: float
    rmse: float
    attack_ms: int
    stop: str
    full_res_miss: bool
    misclassified: bool


@dataclass
class ReporterState:
    pending: dict[str, TaskContext] = field(default_factory=dict)
    task_order: list[str] = field(default_factory=list)
    last_event_ts: float = field(default_factory=time.time)
    last_slack_ts: float = 0.0
    last_heartbeat_ts: float = 0.0
    last_digest_ts: float = field(default_factory=time.time)
    last_task_finished_ts: float | None = None
    digest_records: list[TaskRecord] = field(default_factory=list)


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


def _parse_log_fields(line: str) -> dict[str, str]:
    if "[SMA-MI-FGSM]" not in line:
        return {}
    idx = line.find("[SMA-MI-FGSM]")
    rest = line[idx + len("[SMA-MI-FGSM]") :].strip()
    return {key: value for key, value in RE_KV.findall(rest)}


def _estimate_validator_score(
    *,
    norm: float,
    rmse: float,
    epsilon: float,
    misclassified: bool,
    min_delta: float,
    max_linf_delta: float = DEFAULT_MAX_LINF_DELTA,
    response_time_ms: int = 0,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[float, str]:
    if not misclassified:
        return 0.0, "label_match_with_original"
    if norm < min_delta:
        return 0.0, "below_min_delta"
    effective_max = min(float(epsilon), max_linf_delta)
    if norm > effective_max:
        return 0.0, "above_max_delta"
    denom = max(1e-12, effective_max - min_delta)
    linf_ratio = min(max((norm - min_delta) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_weight = max(1e-12, DEFAULT_LINF_WEIGHT + DEFAULT_RMSE_WEIGHT)
    perturbation_score = (
        (DEFAULT_LINF_WEIGHT * linf_score) + (DEFAULT_RMSE_WEIGHT * rmse_score)
    ) / total_weight
    time_ratio = response_time_ms / (timeout_seconds * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)
    score = perturbation_score + speed_score * 0.0  # speed weight is 0 on subnet today
    return float(score), "success_est"


def _gate_checks(
    *,
    norm: float,
    rmse: float,
    epsilon: float,
    misclassified: bool,
    min_delta: float,
    elapsed_ms: int,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[str]:
    effective_max = min(float(epsilon), DEFAULT_MAX_LINF_DELTA)
    checks: list[str] = []
    checks.append(f"misclassified: {'PASS' if misclassified else 'FAIL'}")
    checks.append(f"min_delta (>={min_delta:.4f}): {'PASS' if norm >= min_delta else 'FAIL'} ({norm:.6f})")
    checks.append(f"epsilon cap (<={effective_max:.4f}): {'PASS' if norm <= effective_max else 'FAIL'} ({norm:.6f})")
    checks.append(f"response_time (<{timeout_seconds}s): {'PASS' if elapsed_ms < timeout_seconds * 1000 else 'WARN'} ({elapsed_ms}ms)")
    checks.append("ssim/psnr: UNKNOWN (validator-only)")
    return checks


def _classify_outcome(*, linf: float, rmse: float, misclassified: bool, elapsed_ms: int, est_score: float) -> tuple[str, str]:
    if not misclassified or linf <= 0.0:
        return "FAIL", "no_effect"
    if elapsed_ms >= 11_000:
        return "WARN", "timeout_risk"
    if est_score >= 0.85:
        return "OK", "good_score"
    if est_score >= 0.5:
        return "OK", "acceptable"
    if misclassified:
        return "LOW", "high_linf_or_rmse"
    return "FAIL", "no_effect"


def _emoji(level: str) -> str:
    return {"OK": ":white_check_mark:", "LOW": ":warning:", "WARN": ":hourglass:", "FAIL": ":x:"}.get(level, ":grey_question:")


def _post_slack(
    webhook_url: str,
    text: str,
    min_interval_s: float,
    state: ReporterState,
    *,
    force: bool = False,
) -> None:
    if not webhook_url:
        return
    now = time.time()
    if not force and now - state.last_slack_ts < min_interval_s:
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


def _fmt_sources(sources: list[str]) -> str:
    if not sources:
        return "none"
    return " → ".join(sources)


def _format_task_message(ctx: TaskContext, *, linf: float, rmse: float, misclassified: bool, state: ReporterState) -> str:
    est_score, est_reason = _estimate_validator_score(
        norm=linf,
        rmse=rmse,
        epsilon=ctx.epsilon,
        misclassified=misclassified,
        min_delta=ctx.min_delta,
        response_time_ms=ctx.elapsed_ms,
    )
    level, reason = _classify_outcome(
        linf=linf,
        rmse=rmse,
        misclassified=misclassified,
        elapsed_ms=ctx.elapsed_ms,
        est_score=est_score,
    )
    inter_task_s = None
    if state.last_task_finished_ts is not None and ctx.forward_ts > 0:
        inter_task_s = int(max(0, ctx.forward_ts - state.last_task_finished_ts))

    title_id = ctx.task_id if len(ctx.task_id) <= 28 else f"{ctx.task_id[:28]}..."
    lines = [
        f"{_emoji(level)} *Perturb miner task* `{title_id}`",
        f"• outcome: *{level}* ({reason})",
        f"• est_validator_score: `{est_score:.4f}` ({est_reason})",
        "",
        "*Challenge*",
        f"• norm: `{ctx.norm_type}` epsilon: `{ctx.epsilon:.6f}` min_delta: `{ctx.min_delta:.6f}`",
    ]
    if ctx.prompt:
        lines.append(f"• prompt: `{ctx.prompt}`")
    if ctx.true_label:
        lines.append(f"• true_label: `{ctx.true_label}`")
    lines.append(
        f"• image: `{ctx.shape or 'unknown'}` work: `{ctx.work_shape or ctx.shape or 'unknown'}` uses_work={ctx.uses_work}"
    )
    if ctx.full_pixels:
        lines.append(f"• full_pixels: `{ctx.full_pixels}`")
    if inter_task_s is not None:
        lines.append(f"• since_last_task: `{inter_task_s}s`")

    lines.extend(
        [
            "",
            "*Attack*",
            f"• algorithm: `{ctx.algorithm or '?'}` ({ctx.algorithm_key or '?'})",
            f"• budget_ms: `{ctx.budget_ms or '?'}` steps: `{ctx.steps or '?'}` restarts_cfg: `{ctx.restarts_cfg or '?'}`",
            f"• attack_ms: `{ctx.elapsed_ms}` remaining_ms: `{ctx.remaining_ms if ctx.remaining_ms is not None else '?'}`",
            f"• stop: `{ctx.stop or '-'}` early_stage: `{ctx.early_stop_stage or '-'}`",
            f"• linf: `{linf:.6f}` rmse: `{rmse:.6f}`",
            f"• label idx: `{ctx.target_idx if ctx.target_idx is not None else '?'}` → `{ctx.final_pred if ctx.final_pred is not None else '?'}` misclassified={misclassified}",
        ]
    )

    if ctx.saliency_ms is not None or ctx.saliency_pixels is not None:
        lines.append(
            f"• saliency: `{ctx.saliency_ms if ctx.saliency_ms is not None else '?'}ms` "
            f"pixels={ctx.saliency_pixels if ctx.saliency_pixels is not None else '?'}"
        )
    if ctx.sparse_result or ctx.sparse_ms is not None:
        lines.append(
            f"• sparse: result=`{ctx.sparse_result or '?'}` "
            f"ms=`{ctx.sparse_ms if ctx.sparse_ms is not None else '?'}` "
            f"topk=`{ctx.sparse_topk if ctx.sparse_topk is not None else '?'}` "
            f"tried=`{ctx.sparse_tried if ctx.sparse_tried is not None else '?'}`"
        )
    if ctx.best_update_sources:
        lines.append(f"• best_updates: `{_fmt_sources(ctx.best_update_sources)}`")
    if ctx.restarts_used:
        lines.append(f"• restarts_used: `{ctx.restarts_used}`")

    if ctx.project_full_success is not None:
        proj = "PASS" if ctx.project_full_success else "FAIL"
        lines.append(
            f"• project_to_full: `{proj}` "
            f"linf=`{ctx.project_full_linf if ctx.project_full_linf is not None else '?'}` "
            f"rmse=`{ctx.project_full_rmse if ctx.project_full_rmse is not None else '?'}`"
        )

    lines.append("")
    lines.append("*Validator gates (miner-side)*")
    for check in _gate_checks(
        norm=linf,
        rmse=rmse,
        epsilon=ctx.epsilon,
        misclassified=misclassified,
        min_delta=ctx.min_delta,
        elapsed_ms=ctx.elapsed_ms,
    ):
        lines.append(f"• {check}")

    if ctx.full_res_miss:
        lines.append("• :warning: *full_res_miss_after_project* (work-scale success did not survive upsample)")
    if level == "WARN":
        lines.append("• :warning: near validator timeout (~15s) — 408 risk")
    if level == "FAIL" and linf <= 0.0:
        lines.append("• validator score likely *0* (no perturbation)")
    elif level == "LOW":
        lines.append("• misclassified but *low score* expected (linf/rmse too high vs epsilon)")
    elif level == "OK" and est_score >= 0.85:
        lines.append("• competitive range if SSIM/PSNR gates pass")

    return "\n".join(lines)


def _record_task(state: ReporterState, *, level: str, est_score: float, ctx: TaskContext, linf: float, rmse: float, misclassified: bool) -> None:
    state.digest_records.append(
        TaskRecord(
            ts=time.time(),
            level=level,
            est_score=est_score,
            linf=linf,
            rmse=rmse,
            attack_ms=ctx.elapsed_ms,
            stop=ctx.stop,
            full_res_miss=ctx.full_res_miss,
            misclassified=misclassified,
        )
    )
    window = float(os.getenv("SLACK_DIGEST_WINDOW_SECONDS", "600"))
    cutoff = time.time() - window
    state.digest_records = [record for record in state.digest_records if record.ts >= cutoff]


def _format_digest(state: ReporterState) -> str:
    records = state.digest_records
    if not records:
        return ":bar_chart: *Perturb digest* — no tasks in window"

    total = len(records)
    by_level = {level: 0 for level in ("OK", "LOW", "WARN", "FAIL")}
    for record in records:
        by_level[record.level] = by_level.get(record.level, 0) + 1

    scored = [record.est_score for record in records if record.est_score > 0]
    avg_score = sum(scored) / len(scored) if scored else 0.0
    avg_ms = sum(record.attack_ms for record in records) / total
    miss = sum(1 for record in records if record.full_res_miss)
    timeout = sum(1 for record in records if record.stop == "time_budget")
    optimal = sum(1 for record in records if record.stop == "optimal_linf")
    flipped = sum(1 for record in records if record.misclassified)

    return (
        f":bar_chart: *Perturb digest* (last {int(os.getenv('SLACK_DIGEST_WINDOW_SECONDS', '600'))}s)\n"
        f"• tasks: `{total}` OK=`{by_level.get('OK', 0)}` LOW=`{by_level.get('LOW', 0)}` "
        f"WARN=`{by_level.get('WARN', 0)}` FAIL=`{by_level.get('FAIL', 0)}`\n"
        f"• label_flips: `{flipped}/{total}` avg_est_score: `{avg_score:.4f}` avg_attack_ms: `{avg_ms:.0f}`\n"
        f"• stop reasons: optimal_linf=`{optimal}` time_budget=`{timeout}` full_res_miss=`{miss}`"
    )


def _handle_sma_line(clean: str, ctx: TaskContext) -> None:
    fields = _parse_log_fields(clean)

    if "phase=saliency_start" in clean:
        if fields.get("elapsed_ms"):
            ctx._saliency_start_ms = int(fields["elapsed_ms"])
        return

    if "phase=saliency_done" in clean:
        if fields.get("saliency_pixels"):
            ctx.saliency_pixels = int(float(fields["saliency_pixels"]))
        if fields.get("elapsed_ms") and ctx._saliency_start_ms is not None:
            ctx.saliency_ms = int(fields["elapsed_ms"]) - ctx._saliency_start_ms
        return

    if "phase=sparse_start" in clean:
        if fields.get("elapsed_ms"):
            ctx._sparse_start_ms = int(fields["elapsed_ms"])
        if fields.get("topk"):
            ctx.sparse_topk = int(fields["topk"])
        elif fields.get("configured_topk"):
            ctx.sparse_topk = int(fields["configured_topk"])
        return

    if "phase=sparse_done" in clean:
        ctx.sparse_result = fields.get("result", ctx.sparse_result)
        if fields.get("elapsed_ms") and ctx._sparse_start_ms is not None:
            ctx.sparse_ms = int(fields["elapsed_ms"]) - ctx._sparse_start_ms
        if fields.get("topk"):
            ctx.sparse_topk = int(fields["topk"])
        return

    if "sparse_stopped" in clean:
        if fields.get("tried"):
            ctx.sparse_tried = int(fields["tried"])
        if fields.get("topk"):
            ctx.sparse_topk = int(fields["topk"])
        return

    if "best_updated" in clean and "source=" in clean:
        source = fields.get("source", "unknown")
        ctx.best_update_sources.append(source)
        if source.startswith("restart_"):
            try:
                ctx.restarts_used = max(ctx.restarts_used, int(source.split("_", 1)[1]) + 1)
            except (IndexError, ValueError):
                pass
        return

    if "early_stop" in clean:
        ctx.early_stop_stage = fields.get("stage", ctx.early_stop_stage)
        return

    if "project_to_full" in clean:
        if "success" in fields:
            ctx.project_full_success = fields["success"] == "True"
        if fields.get("linf"):
            ctx.project_full_linf = float(fields["linf"])
        if fields.get("rmse"):
            ctx.project_full_rmse = float(fields["rmse"])
        if fields.get("full_pixels"):
            ctx.full_pixels = int(fields["full_pixels"])
        return

    if "full_res_miss_after_project" in clean:
        ctx.full_res_miss = True


def _handle_line(line: str, args: argparse.Namespace, state: ReporterState) -> None:
    state.last_event_ts = time.time()
    clean = line.strip()
    if not clean:
        return

    if RE_ERROR.search(clean):
        snippet = clean[-800:] if len(clean) > 800 else clean
        _post_slack(
            args.webhook_url,
            f":rotating_light: *Perturb miner error*\n```{snippet}```",
            args.min_post_interval_s,
            state,
        )
        return

    match = RE_BLACKLIST.search(clean)
    if match:
        _post_slack(
            args.webhook_url,
            f":no_entry: *Blacklist reject*\n• {match.group('reason')}",
            args.min_post_interval_s,
            state,
        )
        return

    match = RE_SKIP.search(clean)
    if match:
        _post_slack(
            args.webhook_url,
            f":fast_forward: *Skipped task* `{match.group('task_id')}`\n• {match.group('reason')}",
            args.min_post_interval_s,
            state,
        )
        return

    match = RE_FORWARD.search(clean)
    if match:
        task_id = match.group("task_id")
        state.pending[task_id] = TaskContext(
            task_id=task_id,
            norm_type=match.group("norm_type"),
            epsilon=float(match.group("epsilon")),
            forward_ts=time.time(),
        )
        state.task_order.append(task_id)
        return

    if "[CHALLENGE_META]" in clean:
        try:
            payload = json.loads(clean.split("[CHALLENGE_META]", 1)[1].strip())
        except (json.JSONDecodeError, IndexError):
            return
        task_id = str(payload.get("task_id", "") or "")
        ctx = state.pending.get(task_id) if task_id else None
        if ctx is None:
            ctx = _latest_task(state)
        if ctx is None and task_id:
            ctx = TaskContext(task_id=task_id, forward_ts=time.time())
            state.pending[task_id] = ctx
            state.task_order.append(task_id)
        if ctx is not None:
            ctx.prompt = str(payload.get("prompt", "") or ctx.prompt)
            ctx.true_label = str(payload.get("true_label", "") or ctx.true_label)
        return

    match = RE_ATTACK_START.search(clean)
    if match:
        ctx = _latest_task(state)
        if ctx is not None:
            ctx.shape = f"{match.group('h')}x{match.group('w')}"
            ctx.work_shape = f"{match.group('wh')}x{match.group('ww')}"
            ctx.uses_work = match.group("uses_work") == "True"
            if match.group("full_pixels"):
                ctx.full_pixels = int(match.group("full_pixels"))
            ctx.target_idx = int(match.group("target_idx"))
            ctx.min_delta = float(match.group("min_delta"))
            ctx.steps = int(match.group("steps"))
            ctx.restarts_cfg = int(match.group("restarts"))
            budget = match.group("budget_ms")
            ctx.budget_ms = None if budget == "unlimited" else int(budget)
        return

    if "[SMA-MI-FGSM]" in clean:
        ctx = _latest_task(state)
        if ctx is not None:
            _handle_sma_line(clean, ctx)
        return

    match = RE_DONE.search(clean)
    if match:
        ctx = _latest_task(state)
        if ctx is not None:
            ctx.stop = match.group("stop")
            ctx.elapsed_ms = int(match.group("elapsed_ms"))
            if match.group("remaining_ms"):
                ctx.remaining_ms = int(match.group("remaining_ms"))
            ctx.final_pred = int(match.group("pred_idx"))
            ctx.target_idx = int(match.group("target_idx"))
            ctx.misclassified_attack = match.group("misclassified") == "True"
            ctx.done_linf = float(match.group("linf"))
            ctx.done_rmse = float(match.group("rmse"))
        return

    match = RE_FINISHED.search(clean)
    if match:
        task_id = match.group("task_id")
        ctx = state.pending.pop(task_id, TaskContext(task_id=task_id))
        if task_id in state.task_order:
            state.task_order.remove(task_id)
        ctx.final_pred = int(match.group("final_pred"))
        ctx.target_idx = int(match.group("target_idx"))
        ctx.algorithm = match.group("algorithm")
        ctx.algorithm_key = match.group("algorithm_key")
        ctx.min_delta = float(match.group("min_delta"))
        linf = float(match.group("linf"))
        rmse = float(match.group("rmse"))
        misclassified = ctx.final_pred != ctx.target_idx
        est_score, _ = _estimate_validator_score(
            norm=linf,
            rmse=rmse,
            epsilon=ctx.epsilon,
            misclassified=misclassified,
            min_delta=ctx.min_delta,
            response_time_ms=ctx.elapsed_ms,
        )
        level, _ = _classify_outcome(
            linf=linf,
            rmse=rmse,
            misclassified=misclassified,
            elapsed_ms=ctx.elapsed_ms,
            est_score=est_score,
        )
        msg = _format_task_message(ctx, linf=linf, rmse=rmse, misclassified=misclassified, state=state)
        _post_slack(args.webhook_url, msg, args.min_post_interval_s, state)
        _record_task(state, level=level, est_score=est_score, ctx=ctx, linf=linf, rmse=rmse, misclassified=misclassified)
        state.last_task_finished_ts = time.time()
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
        f"• last log event: `{idle_s}s` ago\n"
        f"• digest window tasks: `{len(state.digest_records)}`",
        args.min_post_interval_s,
        state,
    )
    state.last_heartbeat_ts = now


def _maybe_digest(args: argparse.Namespace, state: ReporterState) -> None:
    if args.digest_seconds <= 0:
        return
    now = time.time()
    if now - state.last_digest_ts < args.digest_seconds:
        return
    if not state.digest_records:
        state.last_digest_ts = now
        return
    _post_slack(
        args.webhook_url,
        _format_digest(state),
        args.min_post_interval_s,
        state,
        force=True,
    )
    state.last_digest_ts = now


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
    parser.add_argument(
        "--digest-seconds",
        type=int,
        default=int(os.getenv("SLACK_DIGEST_SECONDS", "600")),
        help="Post rolling digest every N seconds (0=off)",
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

    boot_state = ReporterState()
    _post_slack(
        webhook_url,
        f":rocket: *Perturb Slack watcher started*\n"
        f"• tailing `{args.pm2_process}` logs\n"
        f"• digest every `{args.digest_seconds}s` heartbeat every `{args.heartbeat_seconds}s`",
        0.0,
        boot_state,
        force=True,
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
            _maybe_digest(args, state)
            _maybe_heartbeat(args, state)
    except KeyboardInterrupt:
        print("\n[miner_log_reporter] stopped", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
