#!/usr/bin/env python3
"""
orze: GPU experiment orchestrator using filesystem coordination.

Parses ideas from a markdown file, claims them via atomic mkdir,
launches training on free GPUs, and generates a leaderboard.

Usage:
    python farm.py                          # run continuously on all GPUs
    python farm.py --gpus 0,1 --timeout 3600
    python farm.py --once                   # one cycle then exit
    python farm.py --report-only            # just regenerate report.md
"""

import argparse
import datetime
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Ideas parsing
# ---------------------------------------------------------------------------

def parse_ideas(path: str) -> Dict[str, dict]:
    """Parse ideas.md into {idea_id: {title, priority, config, raw}}."""
    text = Path(path).read_text()
    ideas = {}
    # Split on idea headers
    pattern = re.compile(r"^## (idea-\d+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        idea_id = m.group(1)
        title = m.group(2).strip()
        # Extract raw content until next idea or EOF
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end]

        # Extract priority
        pri_match = re.search(r"\*\*Priority\*\*:\s*(\w+)", raw)
        priority = pri_match.group(1).lower() if pri_match else "medium"

        # Extract YAML config block
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL)
        config = {}
        if yaml_match:
            try:
                config = yaml.safe_load(yaml_match.group(1)) or {}
            except yaml.YAMLError:
                pass

        ideas[idea_id] = {
            "title": title,
            "priority": priority,
            "config": config,
            "raw": raw.strip(),
        }
    return ideas


PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def get_unclaimed(ideas: Dict[str, dict], results_dir: Path) -> List[str]:
    """Return idea IDs that have no results directory, sorted by priority then ID."""
    unclaimed = []
    for idea_id, info in ideas.items():
        idea_dir = results_dir / idea_id
        if not idea_dir.exists():
            unclaimed.append(idea_id)

    def sort_key(idea_id):
        pri = PRIORITY_ORDER.get(ideas[idea_id]["priority"], 2)
        # Extract numeric part for stable ordering
        num = int(re.search(r"\d+", idea_id).group())
        return (pri, num)

    unclaimed.sort(key=sort_key)
    return unclaimed


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------

def claim(idea_id: str, results_dir: Path, gpu: int) -> bool:
    """Atomically claim an idea via mkdir. Returns True if we got it."""
    idea_dir = results_dir / idea_id
    try:
        idea_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False

    claim_info = {
        "claimed_by": socket.gethostname(),
        "claimed_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
        "gpu": gpu,
    }
    (idea_dir / "claim.json").write_text(json.dumps(claim_info, indent=2))
    return True


# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------

def get_gpu_memory_used(gpu_id: int) -> Optional[int]:
    """Get GPU memory used in MiB. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def detect_all_gpus() -> List[int]:
    """Auto-detect available GPU indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        return []


def get_free_gpus(gpu_ids: List[int], active: Dict[int, "TrainingProcess"],
                  mem_threshold: int = 2000) -> List[int]:
    """Return GPUs from gpu_ids that are not active and have low memory usage."""
    free = []
    for g in gpu_ids:
        if g in active:
            continue
        mem_used = get_gpu_memory_used(g)
        if mem_used is not None and mem_used > mem_threshold:
            continue
        free.append(g)
    return free


# ---------------------------------------------------------------------------
# Training process management
# ---------------------------------------------------------------------------

@dataclass
class TrainingProcess:
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float


def launch(idea_id: str, gpu: int, results_dir: Path, ideas_md: str,
           config: str, train_script: str, timeout: float) -> TrainingProcess:
    """Launch a training subprocess on the given GPU."""
    log_path = results_dir / idea_id / "train_output.log"

    cmd = [
        sys.executable, train_script,
        "--idea-id", idea_id,
        "--results-dir", str(results_dir),
        "--ideas-md", ideas_md,
        "--config", config,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )

    return TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=time.time(), log_path=log_path, timeout=timeout,
    )


def check_active(active: Dict[int, TrainingProcess],
                 results_dir: Path) -> List[str]:
    """Check running processes. Reap completed/timed-out. Returns list of finished idea IDs."""
    finished = []
    for gpu in list(active.keys()):
        tp = active[gpu]
        ret = tp.process.poll()

        # Still running — check timeout
        if ret is None:
            elapsed = time.time() - tp.start_time
            if elapsed > tp.timeout:
                print(f"  [TIMEOUT] {tp.idea_id} after {elapsed/60:.0f}m — killing")
                tp.process.kill()
                tp.process.wait()
                _write_failure(results_dir / tp.idea_id, "Timed out")
                del active[gpu]
                finished.append(tp.idea_id)
            continue

        # Process exited
        elapsed = time.time() - tp.start_time
        metrics_path = results_dir / tp.idea_id / "metrics.json"

        if ret == 0 and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            status = metrics.get("status", "COMPLETED")
            print(f"  [{status}] {tp.idea_id} on GPU {gpu} in {elapsed/60:.1f}m")
        else:
            reason = f"exit code {ret}"
            # Show last few lines of log for debugging
            try:
                lines = tp.log_path.read_text().strip().split("\n")
                tail = "\n".join(lines[-5:])
                reason += f"\n{tail}"
            except Exception:
                pass
            print(f"  [FAILED] {tp.idea_id} on GPU {gpu} — {reason}")
            if not metrics_path.exists():
                _write_failure(results_dir / tp.idea_id, f"Process exited with code {ret}")

        del active[gpu]
        finished.append(tp.idea_id)

    return finished


def _write_failure(idea_dir: Path, reason: str):
    """Write a failure metrics.json."""
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    (idea_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def update_report(results_dir: Path, ideas: Dict[str, dict]):
    """Generate a leaderboard report.md from all results."""
    rows = []
    for idea_id in sorted(ideas.keys(), key=lambda x: int(re.search(r"\d+", x).group())):
        idea_dir = results_dir / idea_id
        title = ideas[idea_id]["title"]

        if not idea_dir.exists():
            rows.append({"id": idea_id, "title": title, "status": "QUEUED", "metrics": {}})
            continue

        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append({"id": idea_id, "title": title, "status": "IN_PROGRESS", "metrics": {}})
            continue

        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {"status": "FAILED", "error": "corrupt metrics.json"}

        rows.append({
            "id": idea_id, "title": title,
            "status": metrics.get("status", "UNKNOWN"),
            "metrics": metrics,
        })

    # Count statuses
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Orze Report",
        f"**Updated:** {now} | **Host:** {socket.gethostname()}",
        "",
        "## Pipeline Status",
        f"| Total | Completed | Failed | In Progress | Queued |",
        f"|-------|-----------|--------|-------------|--------|",
        f"| {len(rows)} | {counts.get('COMPLETED', 0)} | {counts.get('FAILED', 0)} "
        f"| {counts.get('IN_PROGRESS', 0)} | {counts.get('QUEUED', 0)} |",
        "",
        "## Results",
        "",
    ]

    # Leaderboard: completed ideas sorted by test_accuracy descending
    completed = [r for r in rows if r["status"] == "COMPLETED"]
    completed.sort(key=lambda r: r["metrics"].get("test_accuracy", 0), reverse=True)

    if completed:
        lines.append("| Rank | Idea | Title | Accuracy | Loss | Time (s) |")
        lines.append("|------|------|-------|----------|------|----------|")
        for rank, r in enumerate(completed, 1):
            m = r["metrics"]
            acc = m.get("test_accuracy", 0)
            loss = m.get("test_loss", 0)
            t = m.get("training_time", 0)
            lines.append(
                f"| {rank} | {r['id']} | {r['title'][:40]} | {acc:.4f} | {loss:.4f} | {t:.0f} |"
            )
        lines.append("")

    # Failed ideas
    failed = [r for r in rows if r["status"] == "FAILED"]
    if failed:
        lines.append("## Failed")
        for r in failed:
            err = r["metrics"].get("error", "unknown")
            lines.append(f"- **{r['id']}**: {r['title'][:50]} — {err[:80]}")
        lines.append("")

    # Queue
    queued = [r for r in rows if r["status"] == "QUEUED"]
    if queued:
        lines.append(f"## Queue ({len(queued)} ideas)")
        for r in queued[:20]:  # Show first 20
            pri = ideas[r["id"]]["priority"]
            lines.append(f"- **{r['id']}** [{pri}]: {r['title'][:60]}")
        if len(queued) > 20:
            lines.append(f"- ... and {len(queued) - 20} more")
        lines.append("")

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Report updated: {report_path} ({len(completed)} completed, {len(queued)} queued)")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class Orze:
    def __init__(self, args):
        self.gpu_ids = args.gpus
        self.timeout = args.timeout
        self.poll = args.poll
        self.once = args.once
        self.ideas_md = args.ideas_md
        self.config = args.config
        self.results_dir = Path(args.results_dir)
        self.train_script = args.train_script
        self.active: Dict[int, TrainingProcess] = {}
        self.running = True

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        print(f"\n[orze] Received signal {signum}, shutting down gracefully...")
        self.running = False
        for gpu, tp in self.active.items():
            print(f"  Waiting for {tp.idea_id} on GPU {gpu}...")
            tp.process.wait(timeout=60)
        print("[orze] Shutdown complete.")
        sys.exit(0)

    def run(self):
        print(f"[orze] Starting on GPUs {self.gpu_ids}")
        print(f"[orze] Ideas: {self.ideas_md}")
        print(f"[orze] Results: {self.results_dir}")
        print(f"[orze] Timeout: {self.timeout}s | Poll: {self.poll}s")
        print()

        iteration = 0
        while self.running:
            iteration += 1
            print(f"--- Iteration {iteration} [{datetime.datetime.now().strftime('%H:%M:%S')}] ---")

            # 1. Reap completed processes
            if self.active:
                check_active(self.active, self.results_dir)

            # 2. Parse ideas and find unclaimed
            ideas = parse_ideas(self.ideas_md)
            unclaimed = get_unclaimed(ideas, self.results_dir)

            # 3. Find free GPUs and launch training
            free = get_free_gpus(self.gpu_ids, self.active)

            if unclaimed and free:
                for gpu in free:
                    if not unclaimed:
                        break
                    idea_id = unclaimed[0]
                    if claim(idea_id, self.results_dir, gpu):
                        print(f"  Launching {idea_id} on GPU {gpu}: {ideas[idea_id]['title'][:50]}")
                        tp = launch(
                            idea_id, gpu, self.results_dir,
                            self.ideas_md, self.config, self.train_script, self.timeout,
                        )
                        self.active[gpu] = tp
                        unclaimed.pop(0)
                    else:
                        # Someone else claimed it, skip
                        unclaimed.pop(0)
            elif not unclaimed:
                if not self.active:
                    print("  All ideas completed!")
                    if not self.once:
                        print("  Waiting for new ideas...")
                else:
                    print(f"  No unclaimed ideas. {len(self.active)} training in progress.")
            else:
                print(f"  {len(unclaimed)} ideas queued, no free GPUs "
                      f"({len(self.active)} active)")

            # 4. Update report
            update_report(self.results_dir, ideas)

            if self.once:
                # In --once mode, wait for all active to finish
                if self.active:
                    print("\n[orze] --once mode: waiting for active training to complete...")
                    while self.active:
                        time.sleep(5)
                        check_active(self.active, self.results_dir)
                    # Final report
                    ideas = parse_ideas(self.ideas_md)
                    update_report(self.results_dir, ideas)
                print("[orze] Done.")
                break

            time.sleep(self.poll)

        print(f"[orze] Exited after {iteration} iterations.")


def main():
    parser = argparse.ArgumentParser(
        description="orze: GPU experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python farm.py                           # all GPUs, continuous
  python farm.py --gpus 0,1 --timeout 3600 # 2 GPUs, 1h timeout
  python farm.py --once                    # one cycle then exit
  python farm.py --report-only             # regenerate report.md
        """,
    )
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max training time in seconds (default: 3600)")
    parser.add_argument("--poll", type=int, default=30,
                        help="Seconds between iterations (default: 30)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="Only regenerate report.md")
    parser.add_argument("--ideas-md", type=str, default="ideas.md",
                        help="Path to ideas markdown file")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory for results")
    parser.add_argument("--train-script", type=str, default="train_idea.py",
                        help="Training script to run per idea")
    args = parser.parse_args()

    # Parse GPU list
    if args.gpus:
        args.gpus = [int(x) for x in args.gpus.split(",")]
    else:
        args.gpus = detect_all_gpus()
        if not args.gpus:
            print("No GPUs detected. Use --gpus to specify manually.")
            sys.exit(1)

    # Report-only mode
    if args.report_only:
        ideas = parse_ideas(args.ideas_md)
        update_report(Path(args.results_dir), ideas)
        return

    orze = Orze(args)
    orze.run()


if __name__ == "__main__":
    main()
