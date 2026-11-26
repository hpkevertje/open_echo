"""
recorder.py
-----------
Rolling diagnostic recorder for raw OpenEcho pings.

The recorder runs in the web server, not in the browser, so closing the page
or losing Wi-Fi does not stop it.  Files are gzipped JSON Lines; each line has
the raw spectrogram plus the depth/sediment metadata for one ping.

The data directory is kept under a size cap.  When the cap is reached the
oldest closed segments are removed, so the recorder can be left on
unattended.
"""

import asyncio
import gzip
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from settings import Settings

log = logging.getLogger("uvicorn")

FILE_PREFIX = "openecho_"
ACTIVE_SUFFIX = ".jsonl"
CLOSED_SUFFIX = ".jsonl.gz"


class Recorder:
    def __init__(self):
        self.settings: Optional[Settings] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_task: Optional[asyncio.Task] = None
        self._queue: Optional[asyncio.Queue] = None
        self._file = None
        self._file_path: Optional[Path] = None
        self._file_opened_at = 0.0
        self._running = False
        self._recorded = 0
        self._dropped = 0
        self._last_record_at = 0.0
        self._error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def directory(self) -> Path:
        if self.settings is not None:
            return Path(self.settings.recording_dir).expanduser()
        return Path("recordings")

    # -- settings / lifecycle --------------------------------------------

    async def update_settings(
        self,
        settings: Settings,
        previous: Optional[Settings] = None,
    ) -> None:
        now = time.time()
        self.settings = settings

        if previous is not None and previous.recording_enabled:
            # The config form does not contain these two derived fields.
            if settings.recording_started_at is None:
                settings.recording_started_at = previous.recording_started_at
            if settings.recording_stop_at is None:
                settings.recording_stop_at = previous.recording_stop_at

        if not settings.recording_enabled:
            await self.stop()
            settings.recording_started_at = None
            settings.recording_stop_at = None
            return

        if settings.recording_started_at is None:
            settings.recording_started_at = now

        if settings.recording_duration_hours > 0:
            if (
                previous is None
                or not previous.recording_enabled
                or previous.recording_duration_hours != settings.recording_duration_hours
                or settings.recording_stop_at is None
            ):
                settings.recording_stop_at = (
                    settings.recording_started_at
                    + settings.recording_duration_hours * 3600.0
                )
        else:
            settings.recording_stop_at = None

        if settings.recording_stop_at is not None and now >= settings.recording_stop_at:
            settings.recording_enabled = False
            settings.recording_started_at = None
            settings.recording_stop_at = None
            await self.stop()
            return

        restart = (
            self._running
            and previous is not None
            and previous.recording_dir != settings.recording_dir
        )
        if restart:
            await self.stop()

        if not self._running:
            await self.start()
        elif self.settings is not None:
            self._enforce_retention()

    async def start(self) -> None:
        if self._running or self.settings is None:
            return

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._compress_orphans()
            self._enforce_retention()
            self._queue = asyncio.Queue(maxsize=2000)
            self._error = None
            self._running = True
            self._open_file()

            self._task = asyncio.create_task(self._writer())
            if self.settings.recording_stop_at is not None:
                self._stop_task = asyncio.create_task(self._watch_stop())

            log.info("Recorder started: %s", self._file_path)
        except Exception as e:
            self._running = False
            self._error = str(e)
            self._close_file()
            log.error("Recorder failed to start: %s", e)

    async def stop(self) -> None:
        self._running = False

        if self._stop_task is not None:
            self._stop_task.cancel()
            try:
                await self._stop_task
            except asyncio.CancelledError:
                pass
            self._stop_task = None

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._close_file()
        self._queue = None

    async def _auto_stop(self) -> None:
        """Stop because the configured timer elapsed."""
        if not self._running:
            return

        self._running = False
        if self.settings is not None:
            self.settings.recording_enabled = False
            self.settings.recording_started_at = None
            self.settings.recording_stop_at = None
            try:
                self.settings.save()
            except Exception as e:
                log.error("Recorder could not save timer stop: %s", e)

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._close_file()
        self._queue = None
        log.info("Recorder stopped by timer")

    async def _watch_stop(self) -> None:
        try:
            while self._running:
                stop_at = self.settings.recording_stop_at if self.settings else None
                if stop_at is None:
                    return
                delay = stop_at - time.time()
                if delay <= 0:
                    await self._auto_stop()
                    return
                await asyncio.sleep(min(delay, 5.0))
        except asyncio.CancelledError:
            return

    # -- recording -------------------------------------------------------

    def record(self, data: dict) -> None:
        if not self._running or self.settings is None or self._queue is None:
            return

        now = time.time()
        if self.settings.recording_stop_at is not None and now >= self.settings.recording_stop_at:
            return

        interval = 1.0 / max(float(self.settings.recording_hz), 0.1)
        if now - self._last_record_at < interval:
            return
        self._last_record_at = now

        record = {
            "t": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "ts": now,
            "depth": data.get("measured_depth"),
            "depth_index": data.get("depth_index"),
            "valid": data.get("depth_valid"),
            "held": data.get("depth_held"),
            "confidence": data.get("depth_confidence"),
            "echo_db": data.get("depth_echo_db"),
            "temperature": data.get("temperature"),
            "drive_voltage": data.get("drive_voltage"),
            "resolution": data.get("resolution"),
            "sediment": data.get("sediment"),
            "spectrogram": data.get("spectrogram"),
        }

        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1

    async def _writer(self) -> None:
        try:
            while self._running:
                record = await self._queue.get()
                if not self._running:
                    break
                assert self._file is not None
                line = json.dumps(record, separators=(",", ":"))
                self._file.write(line + "\n")
                self._file.flush()
                self._recorded += 1

                if (
                    time.time() - self._file_opened_at
                    >= self.settings.recording_segment_minutes * 60.0
                ):
                    await self._rotate()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error = str(e)
            self._running = False
            self._close_file()
            log.error("Recorder write failed: %s", e)

    # -- files -----------------------------------------------------------

    def _open_file(self) -> None:
        self._close_file()
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.directory / f"{FILE_PREFIX}{stamp}.jsonl"
        if path.exists():
            path = self.directory / f"{FILE_PREFIX}{stamp}_{int(time.time() * 1000) % 1000:03d}.jsonl"

        # Write plain JSONL while a segment is active so it can be downloaded
        # immediately; compress only when the segment is closed.
        self._file = open(path, "a", encoding="utf-8")
        self._file_path = path
        self._file_opened_at = time.time()
        log.info("Recorder file: %s", path)

    def _compress(self, path: Path) -> None:
        try:
            if not path.exists():
                return
            if path.stat().st_size == 0:
                path.unlink()
                return

            gz_path = path.with_name(path.name + ".gz")
            with open(path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            path.unlink()
            log.info("Recorder compressed: %s", gz_path.name)
        except Exception as e:
            log.error("Recorder could not compress %s: %s", path, e)

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        if self._file_path is not None:
            path = self._file_path
            self._file_path = None
            try:
                if path.exists():
                    if path.stat().st_size == 0:
                        path.unlink()
                    else:
                        self._compress(path)
            except Exception:
                pass

    def _compress_orphans(self) -> None:
        try:
            for path in self.directory.glob(f"{FILE_PREFIX}*.jsonl"):
                if path != self._file_path:
                    self._compress(path)
        except Exception as e:
            log.warning("Recorder could not process orphan files: %s", e)

    async def _rotate(self) -> None:
        # Close the plain active segment, compress it off the event loop, then
        # start the next segment.
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        path = self._file_path
        self._file_path = None
        if path is not None:
            await asyncio.to_thread(self._compress, path)

        self._open_file()
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        if self.settings is None:
            return
        try:
            max_bytes = float(self.settings.recording_max_mb) * 1024.0 * 1024.0
            files = self.list_files()
            total = sum(item["size"] for item in files)
            # Never delete the currently open segment.
            for item in sorted(files, key=lambda x: x["mtime"]):
                if total <= max_bytes:
                    break
                if item["active"]:
                    continue
                path = self.path_for(item["name"])
                if path is None:
                    continue
                try:
                    total -= path.stat().st_size
                    path.unlink()
                    log.info("Recorder removed old segment: %s", path.name)
                except Exception as e:
                    log.warning("Recorder could not remove %s: %s", path, e)
        except Exception as e:
            self._error = str(e)
            log.error("Recorder retention failed: %s", e)

    def list_files(self) -> list[dict]:
        try:
            directory = self.directory
            if not directory.exists():
                return []
            files = []
            paths = list(directory.glob(f"{FILE_PREFIX}*{ACTIVE_SUFFIX}"))
            paths += list(directory.glob(f"{FILE_PREFIX}*{CLOSED_SUFFIX}"))
            for path in paths:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024.0 * 1024.0), 2),
                        "mtime": stat.st_mtime,
                        "active": path == self._file_path,
                    }
                )
            return sorted(files, key=lambda item: item["mtime"], reverse=True)
        except Exception:
            return []

    def path_for(self, name: str) -> Optional[Path]:
        if (
            not name
            or Path(name).name != name
            or not name.endswith((ACTIVE_SUFFIX, CLOSED_SUFFIX))
        ):
            return None
        directory = self.directory.resolve()
        path = (directory / name).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            return None
        return path if path.is_file() else None

    def status(self) -> dict:
        files = self.list_files()
        used = sum(item["size"] for item in files)
        max_mb = float(self.settings.recording_max_mb) if self.settings else 0.0
        return {
            "enabled": self._running,
            "directory": str(self.directory),
            "recorded": self._recorded,
            "dropped": self._dropped,
            "files": len(files),
            "bytes": used,
            "used_mb": round(used / (1024.0 * 1024.0), 1),
            "max_mb": max_mb,
            "current_file": self._file_path.name if self._file_path else None,
            "started_at": self.settings.recording_started_at if self.settings else None,
            "stop_at": self.settings.recording_stop_at if self.settings else None,
            "stop_at_iso": (
                datetime.fromtimestamp(self.settings.recording_stop_at, timezone.utc).isoformat()
                if self.settings and self.settings.recording_stop_at
                else None
            ),
            "error": self._error,
        }
