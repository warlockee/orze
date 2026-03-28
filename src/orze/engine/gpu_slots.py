"""GPU slot manager for multi-job-per-GPU and multi-GPU-per-job scheduling.

Replaces the simple Dict[int, TrainingProcess] with a slot-based system:
- Each GPU has N slots (configurable via slots_per_gpu)
- Small jobs use 1 slot on 1 GPU
- Large jobs can claim 1 slot on multiple GPUs

Implements dict-like interface for backward compatibility.
When slots_per_gpu=1, behavior is identical to the old Dict[int, TrainingProcess].
"""

import logging
from typing import Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger("orze")


class GpuSlotManager:
    """Manages GPU slot allocation for training processes.

    Usage:
        mgr = GpuSlotManager([0, 1, 2, 3], slots_per_gpu=4)
        slot_key = mgr.assign(tp, gpus=[0])       # single-GPU job
        slot_key = mgr.assign(tp, gpus=[0, 1])     # multi-GPU job
        mgr.release(slot_key)

    Dict-like interface (backward compat):
        mgr[slot_key] = tp    # assign
        del mgr[slot_key]     # release
        for key, tp in mgr.items(): ...
        len(mgr)              # number of active processes
    """

    def __init__(self, gpu_ids: List[int], slots_per_gpu: int = 1):
        self.gpu_ids = list(gpu_ids)
        self.slots_per_gpu = max(1, slots_per_gpu)
        # gpu_id -> [slot0_tp, slot1_tp, ...] (None = free)
        self._slots: Dict[int, List] = {
            g: [None] * self.slots_per_gpu for g in self.gpu_ids
        }
        # slot_key -> TrainingProcess
        self._active: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def find_free_slot(self, gpus_required: int = 1,
                       exclude_gpus: Optional[Set[int]] = None,
                       mem_threshold: int = 2000) -> Optional[List[int]]:
        """Find GPU(s) with free slots for a job.

        Returns list of GPU IDs to use, or None if no capacity.
        For gpus_required=1: returns [gpu_id] for a GPU with a free slot.
        For gpus_required=N: returns [gpu_id1, ..., gpu_idN] each with a free slot.
        """
        exclude = exclude_gpus or set()
        gpus_with_slots = []
        for g in self.gpu_ids:
            if g in exclude:
                continue
            if any(s is None for s in self._slots[g]):
                gpus_with_slots.append(g)

        if len(gpus_with_slots) < gpus_required:
            return None

        return gpus_with_slots[:gpus_required]

    def assign(self, tp, gpus: List[int]) -> str:
        """Assign a process to slot(s) on the given GPU(s).
        Returns the slot key."""
        parts = []
        for g in gpus:
            slots = self._slots[g]
            assigned = False
            for i, s in enumerate(slots):
                if s is None:
                    slots[i] = tp
                    parts.append(f"{g}:{i}")
                    assigned = True
                    break
            if not assigned:
                raise RuntimeError(f"No free slot on GPU {g}")

        slot_key = "+".join(parts)
        self._active[slot_key] = tp

        # Set slot_key on tp if it has the attribute
        if hasattr(tp, 'slot_key'):
            tp.slot_key = slot_key

        return slot_key

    def release(self, slot_key: str):
        """Release slot(s) and return the process."""
        tp = self._active.pop(slot_key, None)
        if tp is None:
            return None

        for part in slot_key.split("+"):
            gpu_str, slot_str = part.split(":")
            g, i = int(gpu_str), int(slot_str)
            if g in self._slots and i < len(self._slots[g]):
                self._slots[g][i] = None

        return tp

    def free_gpu_ids(self, exclude: Optional[Set[int]] = None) -> List[int]:
        """GPUs with at least one free slot, sorted by least-loaded first."""
        exc = exclude or set()
        candidates = [(sum(1 for s in self._slots[g] if s is None), g)
                      for g in self.gpu_ids
                      if g not in exc and any(s is None for s in self._slots[g])]
        candidates.sort(key=lambda x: -x[0])  # most free slots first
        return [g for _, g in candidates]

    def gpu_ids_in_use(self) -> Set[int]:
        """GPUs with ANY slot occupied."""
        return {g for g in self.gpu_ids
                if any(s is not None for s in self._slots[g])}

    def fully_busy_gpu_ids(self) -> Set[int]:
        """GPUs with ALL slots occupied."""
        return {g for g in self.gpu_ids
                if all(s is not None for s in self._slots[g])}

    def slot_usage(self, gpu_id: int) -> Tuple[int, int]:
        """Return (used, total) slots for a GPU."""
        slots = self._slots.get(gpu_id, [])
        used = sum(1 for s in slots if s is not None)
        return used, len(slots)

    @property
    def total_free_slots(self) -> int:
        return sum(1 for g in self._slots.values() for s in g if s is None)

    @property
    def total_used_slots(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Dict-compatible interface (backward compat)
    # ------------------------------------------------------------------
    # When slots_per_gpu=1, keys are "G:0" strings. Code that did
    # `for gpu in active.keys()` now gets slot keys instead of ints,
    # but the TrainingProcess.gpu property still works for all reads.

    def keys(self):
        return self._active.keys()

    def values(self):
        return self._active.values()

    def items(self):
        return self._active.items()

    def __len__(self):
        return len(self._active)

    def __contains__(self, key):
        return key in self._active

    def __getitem__(self, key):
        return self._active[key]

    def __setitem__(self, key, tp):
        # For backward compat: active[gpu] = tp
        # If key is an int (old-style), convert to slot key
        if isinstance(key, int):
            slot_key = self.assign(tp, [key])
        else:
            self._active[key] = tp

    def __delitem__(self, key):
        if isinstance(key, int):
            # Find slot key for this GPU
            for sk in list(self._active.keys()):
                if sk.startswith(f"{key}:"):
                    self.release(sk)
                    return
        else:
            self.release(key)

    def __iter__(self):
        return iter(self._active)

    def __bool__(self):
        return bool(self._active)

    def get(self, key, default=None):
        return self._active.get(key, default)
