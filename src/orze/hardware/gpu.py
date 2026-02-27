import subprocess
import logging
from typing import List, Optional

logger = logging.getLogger("orze")


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


def _eval_already_running(idea_id: str, cfg: dict = None) -> bool:
    """Check if an eval process is already running for this idea."""
    eval_script = "eval"
    if cfg:
        script_path = cfg.get("eval_script", "")
        if script_path:
            from pathlib import Path
            eval_script = Path(script_path).stem
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{eval_script}.*{idea_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_all_gpus() -> List[int]:
    """Auto-detect available GPU indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return [int(x.strip()) for x in result.stdout.strip().split("\n")
                if x.strip()]
    except Exception:
        return []


def get_free_gpus(gpu_ids: List[int], active: dict,
                  mem_threshold: int = 2000) -> List[int]:
    """Return GPUs not in active dict and with low memory usage."""
    free = []
    for g in gpu_ids:
        if g in active:
            continue
        mem_used = get_gpu_memory_used(g)
        if mem_used is not None and mem_used > mem_threshold:
            continue
        free.append(g)
    return free


def _query_gpu_details() -> List[dict]:
    """Query nvidia-smi for per-GPU stats (memory, util, temp)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                })
        return gpus
    except Exception:
        return []
