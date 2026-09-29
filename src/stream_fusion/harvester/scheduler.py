"""Homelab Scheduler Engine & Periodic Task Manager (Spec 26).

Coordinates periodic background tasks:
1. Periodic channel crawler (discovers new streamer VODs).
2. Download queue polling and dispatch.
3. Automated FullSpectrumPipeline analysis of harvested VODs.
4. Housekeeping & storage retention cleanup.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.models.schemas import DaemonTaskType, HarvestStatus

logger = logging.getLogger(__name__)


class ScheduledTask:
    """Encapsulates a periodic background task with next-run scheduling."""

    def __init__(
        self,
        name: str,
        task_type: DaemonTaskType,
        interval_seconds: float,
        action: Callable[[], Any],
        enabled: bool = True,
        initial_delay_seconds: float = 0.0,
    ):
        self.name = name
        self.task_type = task_type
        self.interval_seconds = max(1.0, interval_seconds)
        self.action = action
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: datetime = datetime.now(timezone.utc)
        if initial_delay_seconds > 0:
            self.next_run = datetime.fromtimestamp(
                time.time() + initial_delay_seconds, tz=timezone.utc
            )
        self.is_running: bool = False
        self.run_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None

    def is_due(self) -> bool:
        """Checks if the task is due to execute."""
        if not self.enabled or self.is_running:
            return False
        return datetime.now(timezone.utc) >= self.next_run

    def execute(self) -> Any:
        """Executes the task and calculates the next execution timestamp."""
        self.is_running = True
        self.last_run = datetime.now(timezone.utc)
        try:
            result = self.action()
            self.run_count += 1
            self.last_error = None
            return result
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Scheduled task '{self.name}' failed: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False
            self.next_run = datetime.fromtimestamp(
                time.time() + self.interval_seconds, tz=timezone.utc
            )


class HomelabScheduler:
    """Manages collection of scheduled periodic tasks with cooperative execution."""

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self._lock = threading.Lock()

    def add_task(
        self,
        name: str,
        task_type: DaemonTaskType,
        interval_seconds: float,
        action: Callable[[], Any],
        enabled: bool = True,
        initial_delay_seconds: float = 0.0,
    ) -> ScheduledTask:
        """Registers a periodic scheduled task."""
        task = ScheduledTask(
            name=name,
            task_type=task_type,
            interval_seconds=interval_seconds,
            action=action,
            enabled=enabled,
            initial_delay_seconds=initial_delay_seconds,
        )
        with self._lock:
            self.tasks[name] = task
        return task

    def get_task(self, name: str) -> Optional[ScheduledTask]:
        """Retrieves task by name."""
        with self._lock:
            return self.tasks.get(name)

    def trigger_task(self, name: str) -> Any:
        """Immediately executes a task out-of-band and resets its next run timer."""
        task = self.get_task(name)
        if not task:
            raise KeyError(f"Task '{name}' not found.")
        if task.is_running:
            logger.warning(f"Task '{name}' is already running. Skipping trigger.")
            return None
        return task.execute()

    def tick(self) -> List[str]:
        """Checks all registered tasks and executes any that are due."""
        executed: List[str] = []
        with self._lock:
            due_tasks = [t for t in self.tasks.values() if t.is_due()]

        for task in due_tasks:
            try:
                task.execute()
                executed.append(task.name)
            except Exception as e:
                logger.error(f"Error ticking task '{task.name}': {e}")
        return executed


def run_storage_retention_cleanup(
    catalog: HarvestCatalog,
    homelab_root: Path,
    retention_days: int,
) -> int:
    """Prunes raw media.mp4 files for VODs analyzed older than retention_days.
    
    Preserves chat.json, metadata.json, checksums.sha256, and analytical artifacts.
    Returns the count of pruned media files.
    """
    if retention_days <= 0:
        return 0

    now = datetime.now(timezone.utc)
    analyzed_vods = catalog.list_vods(status=HarvestStatus.ANALYZED, limit=500)
    pruned_count = 0

    for vod in analyzed_vods:
        if not vod.analyzed_at or not vod.video_path:
            continue

        # Check age
        age_days = (now - vod.analyzed_at).total_seconds() / 86400.0
        if age_days >= retention_days:
            media_file = Path(vod.video_path)
            if media_file.exists():
                try:
                    media_file.unlink()
                    logger.info(
                        f"Retention policy ({retention_days}d): Pruned raw video {media_file.name} "
                        f"for analyzed VOD {vod.vod_id} (age: {age_days:.1f} days)."
                    )
                    pruned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {media_file}: {e}")

    return pruned_count
