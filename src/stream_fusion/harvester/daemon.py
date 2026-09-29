"""Homelab 24/7 Scheduler Daemon & Background Supervisor (Spec 26).

Manages continuous, unsupervised execution of StreamFusion on a homelab server:
1. Periodic channel crawler for target roster streamers.
2. Concurrent download worker pool dispatch.
3. Automated FullSpectrumPipeline analysis of harvested VODs.
4. Cooperative shutdown, signal handling, PID lockfile management, and telemetry.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import psutil

from stream_fusion.harvester.bridge import ingest_to_pipeline
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.harvester.engine import HomelabHarvester
from stream_fusion.harvester.scheduler import HomelabScheduler, run_storage_retention_cleanup
from stream_fusion.models.schemas import (
    DaemonConfig,
    DaemonJobRecord,
    DaemonState,
    DaemonStatusReport,
    DaemonTaskType,
    FullSpectrumConfig,
    HarvestStatus,
)

logger = logging.getLogger(__name__)


def is_process_running(pid: int) -> bool:
    """Checks if a process with given PID is currently active."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception:
        # Fallback for systems without full psutil permissions
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class HomelabDaemon:
    """Continuous 24/7 background supervisor and scheduler for StreamFusion."""

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        catalog: Optional[HarvestCatalog] = None,
        harvester: Optional[HomelabHarvester] = None,
        crawler: Optional[ChannelVodCrawler] = None,
        pipeline_executor: Optional[Callable[..., Any]] = None,
    ):
        self.config = config or DaemonConfig()
        self.homelab_root = Path(self.config.homelab_root).resolve()
        self.homelab_root.mkdir(parents=True, exist_ok=True)

        self.pid_path = Path(self.config.pid_file).resolve()
        self.log_path = Path(self.config.log_file).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Database & Subsystem instances
        self._own_catalog = catalog is None
        self.catalog = catalog or HarvestCatalog(database_url=self.config.catalog_db_url)
        self.crawler = crawler or ChannelVodCrawler()
        self.harvester = harvester or HomelabHarvester(
            catalog=self.catalog,
            homelab_root=self.homelab_root,
            max_concurrent_workers=self.config.max_concurrent_downloads,
            min_free_disk_gb=self.config.min_free_disk_gb,
        )
        self.pipeline_executor = pipeline_executor or ingest_to_pipeline

        # Scheduling & State
        self.scheduler = HomelabScheduler()
        self.state: DaemonState = DaemonState.STOPPED
        self.started_at: Optional[datetime] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Unpaused initially

        self._lock = threading.Lock()
        self._active_pipeline_vod: Optional[str] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._recent_jobs: List[DaemonJobRecord] = []
        self._recent_errors: List[str] = []

        self._setup_logging()
        self._register_scheduled_tasks()

    def _setup_logging(self) -> None:
        """Sets up file logging for the background daemon."""
        self._file_handler: Optional[logging.FileHandler] = None
        try:
            handler = logging.FileHandler(str(self.log_path), encoding="utf-8")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)
            self._file_handler = handler
        except Exception as e:
            logger.warning(f"Could not initialize file handler at {self.log_path}: {e}")


    def _register_scheduled_tasks(self) -> None:
        """Registers periodic crawl, download poll, pipeline, and housekeeping tasks."""
        # 1. Periodic channel crawl
        crawl_interval_sec = max(60.0, self.config.crawl_interval_minutes * 60.0)
        self.scheduler.add_task(
            name="channel_crawl",
            task_type=DaemonTaskType.CRAWL,
            interval_seconds=crawl_interval_sec,
            action=self._scheduled_crawl,
            initial_delay_seconds=5.0,  # Wait 5 seconds after startup before initial crawl
        )

        # 2. Download queue check
        poll_interval_sec = max(2.0, float(self.config.download_poll_interval_seconds))
        self.scheduler.add_task(
            name="queue_downloads",
            task_type=DaemonTaskType.DOWNLOAD,
            interval_seconds=poll_interval_sec,
            action=self._scheduled_downloads,
            initial_delay_seconds=2.0,
        )

        # 3. Auto-Pipeline analysis (if enabled)
        if self.config.auto_analyze:
            self.scheduler.add_task(
                name="auto_pipeline",
                task_type=DaemonTaskType.PIPELINE,
                interval_seconds=10.0,
                action=self._scheduled_pipeline,
                initial_delay_seconds=10.0,
            )

        # 4. Storage retention housekeeping (if configured)
        if self.config.retention_days and self.config.retention_days > 0:
            self.scheduler.add_task(
                name="retention_housekeeping",
                task_type=DaemonTaskType.HOUSEKEEPING,
                interval_seconds=21600.0,  # Run every 6 hours
                action=self._scheduled_housekeeping,
                initial_delay_seconds=60.0,
            )

    # -------------------------------------------------------------------------
    # PID Lockfile Management
    # -------------------------------------------------------------------------

    def acquire_pid_lock(self) -> None:
        """Acquires lockfile by writing current process ID. Fails if another daemon is running."""
        if self.pid_path.exists():
            try:
                with open(self.pid_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        existing_pid = int(content)
                        if is_process_running(existing_pid) and existing_pid != os.getpid():
                            raise RuntimeError(
                                f"Another StreamFusion daemon is already running (PID: {existing_pid})."
                            )
            except (ValueError, OSError) as e:
                logger.warning(f"Overwriting stale or invalid PID lockfile at {self.pid_path}: {e}")

        with open(self.pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        logger.info(f"Acquired PID lockfile at {self.pid_path} with PID {os.getpid()}")

    def release_pid_lock(self) -> None:
        """Removes the PID lockfile if it belongs to this process."""
        if not self.pid_path.exists():
            return
        should_unlink = False
        try:
            with open(self.pid_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and int(content) == os.getpid():
                    should_unlink = True
        except Exception as e:
            logger.warning(f"Error reading PID lockfile: {e}")

        if should_unlink:
            try:
                self.pid_path.unlink(missing_ok=True)
                logger.info(f"Released PID lockfile at {self.pid_path}")
            except Exception as e:
                logger.warning(f"Failed to release PID lockfile: {e}")


    # -------------------------------------------------------------------------
    # Scheduled Task Implementations
    # -------------------------------------------------------------------------

    def _scheduled_crawl(self) -> Dict[str, Any]:
        """Crawl target streamer channels and queue discovered VODs."""
        if not self._pause_event.is_set():
            return {"skipped": "paused"}

        job = self._create_job(DaemonTaskType.CRAWL)
        try:
            logger.info("Executing scheduled channel crawl for target roster...")
            results = self.crawler.sync_all(catalog=self.catalog, auto_queue=True)
            total_discovered = sum(len(vods) for vods in results.values())
            self._finish_job(
                job,
                details={
                    "total_discovered": total_discovered,
                    "streamers_scanned": len(results),
                },
            )
            logger.info(f"Scheduled crawl finished: {total_discovered} new VOD(s) discovered and queued.")
            return {"total_discovered": total_discovered}
        except Exception as e:
            self._record_error(f"Crawl error: {e}")
            self._finish_job(job, error=str(e))
            raise

    def _scheduled_downloads(self) -> Dict[str, Any]:
        """Process download queue using harvester thread pool."""
        if not self._pause_event.is_set():
            return {"skipped": "paused"}

        queued_count = self.catalog.count_vods_by_status().get(HarvestStatus.QUEUED.value, 0)
        if queued_count == 0:
            return {"processed": 0}

        job = self._create_job(DaemonTaskType.DOWNLOAD)
        try:
            logger.info(f"Processing download queue ({queued_count} VODs waiting)...")
            harvested = self.harvester.process_queue()
            self._finish_job(job, details={"harvested_count": len(harvested)})
            return {"harvested": len(harvested)}
        except Exception as e:
            self._record_error(f"Download error: {e}")
            self._finish_job(job, error=str(e))
            raise

    def _scheduled_pipeline(self) -> Dict[str, Any]:
        """Dispatches next harvested VOD into FullSpectrumPipeline with single concurrency."""
        if not self._pause_event.is_set():
            return {"skipped": "paused"}

        with self._lock:
            if self._active_pipeline_vod is not None:
                # Pipeline already actively processing a VOD
                return {"active": self._active_pipeline_vod}

        # Query candidate VODs ready for analysis
        ready_vods = self.catalog.list_vods(status=HarvestStatus.HARVESTED, limit=5)
        if not ready_vods:
            ready_vods = self.catalog.list_vods(status=HarvestStatus.READY_FOR_ANALYSIS, limit=5)

        if not ready_vods:
            return {"ready": 0}

        target_vod = ready_vods[0]

        # Launch pipeline in worker thread to prevent blocking scheduler tick
        self._launch_pipeline_async(target_vod.vod_id)
        return {"launched": target_vod.vod_id}

    def _scheduled_housekeeping(self) -> Dict[str, Any]:
        """Run storage retention cleanup for old analyzed media."""
        if not self.config.retention_days:
            return {"skipped": "no_retention_policy"}

        job = self._create_job(DaemonTaskType.HOUSEKEEPING)
        try:
            logger.info(f"Running housekeeping retention pruning ({self.config.retention_days} days)...")
            pruned = run_storage_retention_cleanup(
                catalog=self.catalog,
                homelab_root=self.homelab_root,
                retention_days=self.config.retention_days,
            )
            self._finish_job(job, details={"pruned_media_files": pruned})
            return {"pruned": pruned}
        except Exception as e:
            self._record_error(f"Housekeeping error: {e}")
            self._finish_job(job, error=str(e))
            raise

    # -------------------------------------------------------------------------
    # Pipeline Execution Runner
    # -------------------------------------------------------------------------

    def _launch_pipeline_async(self, vod_id: str) -> None:
        """Launches pipeline worker thread for a harvested VOD."""
        with self._lock:
            self._active_pipeline_vod = vod_id

        def _worker():
            job = self._create_job(DaemonTaskType.PIPELINE, target_id=vod_id)
            try:
                logger.info(f"Starting FullSpectrumPipeline for harvested VOD {vod_id}...")
                fs_config = FullSpectrumConfig(
                    enable_short_production=self.config.enable_shorts,
                    enable_web_grounding=self.config.enable_web_grounding,
                    enable_adaptive_slang=self.config.enable_adaptive_slang,
                    enable_sponsor_quantifier=self.config.enable_sponsor_quantifier,
                    dry_run_shorts=self.config.dry_run_shorts,
                )
                self.pipeline_executor(
                    vod_id=vod_id,
                    catalog=self.catalog,
                    config=fs_config,
                )
                self._finish_job(job, details={"status": "ANALYZED"})
                logger.info(f"FullSpectrumPipeline completed successfully for {vod_id}!")
            except Exception as e:
                logger.error(f"FullSpectrumPipeline failed for {vod_id}: {e}", exc_info=True)
                self._record_error(f"Pipeline failure ({vod_id}): {e}")
                self._finish_job(job, error=str(e))
                self.catalog.update_vod_status(
                    vod_id, HarvestStatus.ERROR, error_message=f"Pipeline error: {e}"
                )
            finally:
                with self._lock:
                    self._active_pipeline_vod = None

        thread = threading.Thread(target=_worker, name=f"pipeline-{vod_id}", daemon=True)
        self._pipeline_thread = thread
        thread.start()

    # -------------------------------------------------------------------------
    # Lifecycle & Loop
    # -------------------------------------------------------------------------

    def start(self, foreground: bool = True) -> None:
        """Starts daemon supervisor loop."""
        self.acquire_pid_lock()
        self._install_signals()
        self.state = DaemonState.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self._stop_event.clear()
        logger.info(f"StreamFusion Homelab Daemon started (PID: {os.getpid()})")

        if foreground:
            self._run_loop()
        else:
            self._worker_thread = threading.Thread(target=self._run_loop, name="daemon-supervisor", daemon=True)
            self._worker_thread.start()

    def _run_loop(self) -> None:
        """Main event tick loop."""
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    try:
                        self.scheduler.tick()
                    except Exception as e:
                        logger.error(f"Unhandled error in scheduler tick: {e}", exc_info=True)
                        self._record_error(f"Scheduler tick error: {e}")

                # Sleep in short increments for responsive termination
                self._stop_event.wait(timeout=1.0)

        finally:
            self._drain_and_shutdown()

    def stop(self) -> None:
        """Signals daemon to stop gracefully."""
        logger.info("Graceful shutdown requested for StreamFusion Homelab Daemon...")
        self.state = DaemonState.DRAINING
        self._stop_event.set()

    def _drain_and_shutdown(self) -> None:
        """Waits for active pipeline or download tasks before final exit."""
        self.state = DaemonState.DRAINING
        logger.info("Draining active tasks...")

        # Wait up to 10 seconds for active pipeline thread to finish if running
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            logger.info("Waiting for active pipeline execution to checkpoint/finish...")
            self._pipeline_thread.join(timeout=10.0)

        self.release_pid_lock()
        self.close()
        self.state = DaemonState.STOPPED
        logger.info("StreamFusion Homelab Daemon stopped cleanly.")

    def close(self) -> None:
        """Closes open file logging handlers and database connections."""
        if self._file_handler:
            try:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
            except Exception:
                pass
            self._file_handler = None

        if self._own_catalog and self.catalog:
            try:
                self.catalog.close()
            except Exception:
                pass


    def pause(self) -> None:
        """Pauses scheduled background tasks without stopping the supervisor."""
        self._pause_event.clear()
        self.state = DaemonState.PAUSED
        logger.info("StreamFusion Homelab Daemon PAUSED.")

    def resume(self) -> None:
        """Resumes scheduled background tasks."""
        self._pause_event.set()
        self.state = DaemonState.RUNNING
        logger.info("StreamFusion Homelab Daemon RESUMED.")

    def trigger_crawl(self) -> Any:
        """Immediately executes an on-demand crawl."""
        return self.scheduler.trigger_task("channel_crawl")

    def trigger_pipeline(self, vod_id: str) -> bool:
        """Immediately triggers pipeline execution for a specific VOD."""
        with self._lock:
            if self._active_pipeline_vod is not None:
                raise RuntimeError(f"Pipeline already busy with VOD '{self._active_pipeline_vod}'")
        self._launch_pipeline_async(vod_id)
        return True

    # -------------------------------------------------------------------------
    # Telemetry & Status
    # -------------------------------------------------------------------------

    def get_status(self) -> DaemonStatusReport:
        """Returns real-time operational status and metrics."""
        counts = self.catalog.count_vods_by_status()
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds() if self.started_at else 0.0

        crawl_task = self.scheduler.get_task("channel_crawl")
        last_crawl = crawl_task.last_run if crawl_task else None
        next_crawl = crawl_task.next_run if crawl_task else None

        harvester_status = self.harvester.get_status()

        with self._lock:
            active_pipeline = self._active_pipeline_vod
            recent_jobs_copy = list(self._recent_jobs)
            recent_errors_copy = list(self._recent_errors)

        return DaemonStatusReport(
            state=self.state,
            pid=os.getpid() if self.state != DaemonState.STOPPED else None,
            uptime_seconds=round(uptime, 1),
            started_at=self.started_at,
            last_crawl_at=last_crawl,
            next_crawl_at=next_crawl,
            active_downloads=harvester_status.active_downloads,
            active_pipeline_vod=active_pipeline,
            queued_vods_count=counts.get(HarvestStatus.QUEUED.value, 0),
            downloading_vods_count=counts.get(HarvestStatus.DOWNLOADING.value, 0),
            harvested_vods_count=counts.get(HarvestStatus.HARVESTED.value, 0)
            + counts.get(HarvestStatus.READY_FOR_ANALYSIS.value, 0),
            analyzed_vods_count=counts.get(HarvestStatus.ANALYZED.value, 0),
            error_vods_count=counts.get(HarvestStatus.ERROR.value, 0),
            free_disk_gb=harvester_status.free_disk_gb,
            recent_errors=recent_errors_copy,
            recent_jobs=recent_jobs_copy[-10:],
        )

    # Aliases for API consistency
    get_status_report = get_status
    trigger_immediate_crawl = trigger_crawl

    @property
    def status(self) -> DaemonState:
        """Returns the current DaemonState."""
        return self.state

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _install_signals(self) -> None:
        """Registers cooperative signal handlers."""
        def _handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
            if hasattr(signal, "SIGBREAK"):  # Windows CTRL_BREAK
                signal.signal(signal.SIGBREAK, _handler)
        except (ValueError, AttributeError):
            # Happens when not running in main thread (e.g. in test fixtures)
            pass

    def _create_job(self, task_type: DaemonTaskType, target_id: Optional[str] = None) -> DaemonJobRecord:
        job = DaemonJobRecord(
            task_type=task_type,
            target_id=target_id,
            status="RUNNING",
            started_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._recent_jobs.append(job)
            if len(self._recent_jobs) > 50:
                self._recent_jobs.pop(0)
        return job

    def _finish_job(
        self,
        job: DaemonJobRecord,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        job.completed_at = now
        job.duration_sec = round((now - job.started_at).total_seconds(), 2)
        if error:
            job.status = "FAILED"
            job.error_message = error
        else:
            job.status = "COMPLETED"
        if details:
            job.details.update(details)

    def _record_error(self, err: str) -> None:
        timestamped_err = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {err}"
        with self._lock:
            self._recent_errors.append(timestamped_err)
            if len(self._recent_errors) > 20:
                self._recent_errors.pop(0)
