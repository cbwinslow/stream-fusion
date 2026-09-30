"""Unit tests for Storage Guardian, Video Archiver, and Google Drive Offloader (Spec 32)."""

import os
from pathlib import Path
import pytest
import time

from stream_fusion.models.schemas import StorageOverflowPolicy
from stream_fusion.storage.archiver import VideoArchiver
from stream_fusion.storage.gdrive import GoogleDriveOffloader
from stream_fusion.storage.guardian import InsufficientStorageError, StorageGuardian


def test_storage_guardian_report(tmp_path):
    """Verifies that StorageGuardian calculates directory sizes and alert levels."""
    storage_root = tmp_path / "stream_storage"
    storage_root.mkdir()

    # Create dummy proxy file (10MB)
    proxy_file = storage_root / "stream1_proxy.mp4"
    proxy_file.write_bytes(b"0" * (10 * 1024 * 1024))

    # Create dummy short file (2MB)
    shorts_dir = storage_root / "shorts"
    shorts_dir.mkdir()
    (shorts_dir / "short_1.mp4").write_bytes(b"0" * (2 * 1024 * 1024))

    guardian = StorageGuardian(
        storage_root=storage_root,
        budget_gb=100.0,
        min_free_disk_gb=10.0,
        warning_threshold=0.80,
        critical_threshold=0.90,
    )

    report = guardian.get_storage_report(target_dir=storage_root)
    assert report.total_used_bytes == 12 * 1024 * 1024
    assert report.vod_proxies_count == 1
    assert report.shorts_count == 1
    assert report.alert_level == "NORMAL"


def test_storage_guardian_assert_can_allocate(tmp_path):
    """Verifies that assert_can_allocate rejects allocations exceeding the budget."""
    storage_root = tmp_path / "stream_storage"
    storage_root.mkdir()

    guardian = StorageGuardian(
        storage_root=storage_root,
        budget_gb=10.0,  # 10 GB budget
        min_free_disk_gb=5.0,
    )

    # 2 GB allocation should succeed
    assert guardian.assert_can_allocate(required_gb=2.0, target_dir=storage_root) is True

    # 15 GB allocation should exceed 10 GB budget and raise InsufficientStorageError
    with pytest.raises(InsufficientStorageError):
        guardian.assert_can_allocate(required_gb=15.0, target_dir=storage_root)


def test_storage_guardian_prune_oldest_proxies(tmp_path):
    """Verifies FIFO pruning of oldest archival proxies."""
    storage_root = tmp_path / "stream_storage"
    storage_root.mkdir()

    # Create 3 proxies with different mtimes
    p1 = storage_root / "vod_01_proxy.mp4"
    p1.write_bytes(b"A" * (5 * 1024 * 1024))
    os.utime(p1, (time.time() - 300, time.time() - 300))

    p2 = storage_root / "vod_02_proxy.mp4"
    p2.write_bytes(b"B" * (5 * 1024 * 1024))
    os.utime(p2, (time.time() - 200, time.time() - 200))

    p3 = storage_root / "vod_03_proxy.mp4"
    p3.write_bytes(b"C" * (5 * 1024 * 1024))
    os.utime(p3, (time.time() - 100, time.time() - 100))

    guardian = StorageGuardian(storage_root=storage_root, budget_gb=100.0)

    # Free 6 MB (should prune p1 and p2)
    pruned = guardian.prune_oldest_proxies(target_bytes_to_free=6 * 1024 * 1024)
    assert len(pruned) == 2
    assert p1 in pruned
    assert p2 in pruned
    assert not p1.exists()
    assert not p2.exists()
    assert p3.exists()


def test_video_archiver_dry_run(tmp_path):
    """Verifies VideoArchiver dry_run returns simulated 95% reduction."""
    archiver = VideoArchiver()
    source_file = tmp_path / "test_raw.mp4"
    source_file.write_bytes(b"X" * (100 * 1024 * 1024))  # 100 MB

    res = archiver.transcode_to_reference_proxy(
        source_video=source_file,
        dry_run=True,
    )
    assert res.success is True
    assert res.reduction_pct == 95.0
    assert res.original_size_bytes == 100 * 1024 * 1024


def test_video_archiver_purge_ephemeral(tmp_path):
    """Verifies that purge_ephemeral_assets unlinks temporary files and directories."""
    archiver = VideoArchiver()
    temp_wav = tmp_path / "temp_audio_16k.wav"
    temp_wav.write_bytes(b"WAV_DATA" * 1000)

    temp_frames = tmp_path / "temp_keyframes"
    temp_frames.mkdir()
    (temp_frames / "frame_001.jpg").write_bytes(b"JPG_DATA" * 500)
    (temp_frames / "frame_002.jpg").write_bytes(b"JPG_DATA" * 500)

    freed = archiver.purge_ephemeral_assets([temp_wav, temp_frames])
    assert freed > 0
    assert not temp_wav.exists()
    assert not temp_frames.exists()


def test_gdrive_offloader_simulation(tmp_path):
    """Verifies GoogleDriveOffloader computes sha256 and handles simulated cloud upload."""
    offloader = GoogleDriveOffloader(root_folder_id="streamfusion_vault")
    test_proxy = tmp_path / "stream_proxy.mp4"
    test_proxy.write_bytes(b"TEST_PROXY_VIDEO_DATA")

    res = offloader.upload_file(
        file_path=test_proxy,
        mock=True,
        purge_local_on_success=True,
    )

    assert res["uploaded"] is True
    assert res["folder_id"] == "streamfusion_vault"
    assert "sha256" in res
    assert not test_proxy.exists()  # Verified local purge
