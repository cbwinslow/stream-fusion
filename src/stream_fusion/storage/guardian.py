"""Storage Guardian & Budgeting Engine (Spec 32).

Prevents filesystem exhaustion, enforces storage limits, and manages retention lifecycles.
"""

import logging
import os
from pathlib import Path
import shutil
from typing import List, Optional

from stream_fusion.models.schemas import StorageOverflowPolicy, StorageReport

logger = logging.getLogger(__name__)


class InsufficientStorageError(Exception):
    """Raised when an operation would exceed storage budgets or filesystem safety floors."""
    pass


class StorageGuardian:
    """Enforces storage budgets and disk safety floors across StreamFusion workloads."""

    def __init__(
        self,
        storage_root: Path,
        budget_gb: float = 1500.0,
        min_free_disk_gb: float = 5.0,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.90,
        overflow_policy: StorageOverflowPolicy = StorageOverflowPolicy.HALT,
    ):
        self.storage_root = Path(storage_root)
        self.budget_gb = float(budget_gb)
        self.min_free_disk_gb = float(min_free_disk_gb)
        self.warning_threshold = float(warning_threshold)
        self.critical_threshold = float(critical_threshold)
        self.overflow_policy = overflow_policy

    def get_storage_report(self, target_dir: Optional[Path] = None) -> StorageReport:
        """Calculates current disk usage, budget consumption, and alert status."""
        root = Path(target_dir or self.storage_root)
        root.mkdir(parents=True, exist_ok=True)

        # 1. Total bytes consumed by StreamFusion root
        total_used_bytes = 0
        vod_proxies_count = 0
        shorts_count = 0

        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                fp = Path(dirpath) / f
                try:
                    fsize = fp.stat().st_size
                    total_used_bytes += fsize
                    if "proxy" in f.lower() and f.endswith(".mp4"):
                        vod_proxies_count += 1
                    elif "short" in f.lower() or "shorts" in dirpath.lower():
                        shorts_count += 1
                except (OSError, PermissionError):
                    continue

        # 2. Host filesystem free disk space
        try:
            stat = shutil.disk_usage(root)
            free_disk_bytes = stat.free
        except (OSError, ValueError):
            free_disk_bytes = 100 * (1024**3)  # Fallback to 100GB if unavailable

        free_disk_gb = free_disk_bytes / (1024**3)
        budget_bytes = int(self.budget_gb * (1024**3))
        budget_usage_pct = (total_used_bytes / budget_bytes * 100.0) if budget_bytes > 0 else 0.0

        # 3. Determine Alert Level
        if free_disk_gb <= self.min_free_disk_gb or budget_usage_pct >= 95.0:
            alert_level = "HALT"
        elif budget_usage_pct >= (self.critical_threshold * 100.0):
            alert_level = "CRITICAL"
        elif budget_usage_pct >= (self.warning_threshold * 100.0):
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        return StorageReport(
            storage_root=str(root),
            total_used_bytes=total_used_bytes,
            budget_bytes=budget_bytes,
            free_disk_bytes=free_disk_bytes,
            budget_usage_pct=round(budget_usage_pct, 2),
            free_disk_gb=round(free_disk_gb, 2),
            alert_level=alert_level,
            vod_proxies_count=vod_proxies_count,
            shorts_count=shorts_count,
        )

    def assert_can_allocate(self, required_gb: float, target_dir: Optional[Path] = None) -> bool:
        """Verifies if an operation requiring `required_gb` is safe to proceed."""
        report = self.get_storage_report(target_dir)
        required_bytes = int(required_gb * (1024**3))
        projected_used_bytes = report.total_used_bytes + required_bytes
        projected_free_disk_gb = (report.free_disk_bytes - required_bytes) / (1024**3)

        # Check safety floor
        if projected_free_disk_gb < self.min_free_disk_gb:
            msg = (
                f"Storage allocation of {required_gb:.2f} GB rejected: would reduce host free space "
                f"to {projected_free_disk_gb:.2f} GB (minimum safety floor is {self.min_free_disk_gb:.2f} GB)."
            )
            logger.error(msg)
            if self.overflow_policy == StorageOverflowPolicy.PRUNE_OLDEST:
                freed = self.prune_oldest_proxies(required_bytes)
                if freed:
                    return True
            raise InsufficientStorageError(msg)

        # Check budget limit
        if projected_used_bytes > report.budget_bytes:
            msg = (
                f"Storage allocation of {required_gb:.2f} GB rejected: projected usage "
                f"({projected_used_bytes / (1024**3):.2f} GB) exceeds configured budget "
                f"({self.budget_gb:.2f} GB)."
            )
            logger.error(msg)
            if self.overflow_policy == StorageOverflowPolicy.PRUNE_OLDEST:
                freed = self.prune_oldest_proxies(required_bytes)
                if freed:
                    return True
            raise InsufficientStorageError(msg)

        return True

    def prune_oldest_proxies(self, target_bytes_to_free: int) -> List[Path]:
        """Prunes oldest archival proxies until target_bytes_to_free is met."""
        logger.warning("Initiating FIFO proxy pruning to free %d bytes", target_bytes_to_free)
        proxies: List[Path] = []
        for p in self.storage_root.rglob("*.mp4"):
            if "proxy" in p.name.lower() and p.is_file():
                proxies.append(p)

        # Sort oldest first by mtime
        proxies.sort(key=lambda p: p.stat().st_mtime)

        pruned: List[Path] = []
        bytes_freed = 0
        for proxy in proxies:
            if bytes_freed >= target_bytes_to_free:
                break
            try:
                size = proxy.stat().st_size
                proxy.unlink()
                bytes_freed += size
                pruned.append(proxy)
                logger.info("Pruned old proxy %s (freed %d bytes)", proxy.name, size)
            except OSError as ex:
                logger.warning("Failed to prune proxy %s: %s", proxy, ex)

        return pruned
