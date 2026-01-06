"""
Progressive shard downloader for streaming training data.

This module implements a background downloader that maintains a buffer of shards
ahead of the current training position, downloading them progressively to save disk space.
"""

import os
import time
import shutil
import logging
import threading
from typing import Optional, Set, Dict
import requests

from nanochat.common import get_base_dir

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('StreamingDownloader')


class StreamingDownloader:
    """
    Downloads shards progressively in the background, maintaining a buffer of N shards ahead.

    This downloader:
    - Downloads an initial buffer of shards at startup
    - Continuously downloads shards to maintain the buffer as training progresses
    - Blocks training if a needed shard is not yet available
    - Provides status information about download progress
    """

    def __init__(
        self,
        total_shards: int,
        buffer_size: int = 5,
        data_dir: Optional[str] = None,
        num_workers: int = 4,
        base_url: str = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    ):
        """
        Args:
            total_shards: Total number of shards needed for training
            buffer_size: Number of shards to maintain ahead (default: 5)
            data_dir: Directory where shards are saved (default: ~/.cache/nanochat/base_data)
            num_workers: Number of parallel download threads (default: 4)
            base_url: Base URL for downloading shards
        """
        self.total_shards = total_shards
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.base_url = base_url

        # Set up data directory
        if data_dir is None:
            base_dir = get_base_dir()
            data_dir = os.path.join(base_dir, "base_data")
        self.data_dir = os.path.expanduser(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        # Thread-safe state tracking
        self._lock = threading.Lock()
        self._downloaded_shards: Set[int] = set()  # Shards fully downloaded
        self._downloading_shards: Set[int] = set()  # Shards currently being downloaded
        self._current_shard_needed = 0  # The shard currently needed by the dataloader

        # Events for signaling shard availability
        self._shard_events: Dict[int, threading.Event] = {}

        # Download thread control
        self._download_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[Exception] = None

        # Scan for already downloaded shards
        self._scan_existing_shards()

        logger.info(f"Initialized StreamingDownloader: {total_shards} shards, buffer_size={buffer_size}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Found {len(self._downloaded_shards)} shards already downloaded")

    def _scan_existing_shards(self):
        """Scan the data directory for already downloaded shards."""
        if not os.path.exists(self.data_dir):
            return

        for filename in os.listdir(self.data_dir):
            if filename.startswith("shard_") and filename.endswith(".parquet"):
                # Extract shard_id from filename (e.g., "shard_00042.parquet" -> 42)
                try:
                    shard_id = int(filename[6:11])  # Extract the 5-digit number
                    # Verify the file is valid (not corrupted)
                    filepath = os.path.join(self.data_dir, filename)
                    if self._is_valid_shard(filepath):
                        self._downloaded_shards.add(shard_id)
                    else:
                        logger.warning(f"Removing corrupted shard: {filename}")
                        os.remove(filepath)
                except (ValueError, IndexError):
                    logger.warning(f"Skipping invalid filename: {filename}")

    def _is_valid_shard(self, filepath: str) -> bool:
        """Check if a shard file is valid (not corrupted)."""
        if not os.path.exists(filepath):
            return False

        # Check file size (should be at least 1MB for a valid parquet file)
        file_size = os.path.getsize(filepath)
        if file_size < 1024 * 1024:  # Less than 1MB
            return False

        # Try to open with pyarrow to verify it's a valid parquet file
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(filepath)
            # Just opening is enough to verify basic validity
            return True
        except Exception as e:
            logger.warning(f"Invalid parquet file {filepath}: {e}")
            return False

    def start(self):
        """Start the background download thread."""
        if self._download_thread is not None:
            logger.warning("Download thread already running")
            return

        logger.info(f"Starting download of {self.total_shards} shards with buffer_size={self.buffer_size}")

        # Start the download thread
        self._download_thread = threading.Thread(target=self._download_loop, daemon=True)
        self._download_thread.start()

        # Download initial buffer before returning
        logger.info(f"Downloading initial buffer: shards 0-{self.buffer_size - 1}")
        self._wait_for_initial_buffer()
        logger.info("Initial buffer ready, training can start")

    def _wait_for_initial_buffer(self):
        """Wait for the initial buffer to be downloaded."""
        for shard_id in range(min(self.buffer_size, self.total_shards)):
            if not self.wait_for_shard(shard_id, timeout=600.0):  # 10 minutes timeout
                raise RuntimeError(f"Timeout waiting for initial buffer shard {shard_id}")

    def _download_loop(self):
        """Main download loop running in background thread."""
        try:
            while not self._stop_event.is_set():
                # Determine which shards need to be downloaded
                shards_to_download = self._get_shards_to_download()

                if not shards_to_download:
                    # Nothing to download, wait a bit
                    time.sleep(1.0)
                    continue

                # Download shards (sequentially for simplicity in step 1)
                for shard_id in shards_to_download:
                    if self._stop_event.is_set():
                        break

                    # Skip if already downloaded or currently downloading
                    with self._lock:
                        if shard_id in self._downloaded_shards or shard_id in self._downloading_shards:
                            continue
                        self._downloading_shards.add(shard_id)

                    # Download the shard
                    try:
                        self._download_shard(shard_id)
                        with self._lock:
                            self._downloading_shards.discard(shard_id)
                            self._downloaded_shards.add(shard_id)
                            # Signal that this shard is now available
                            if shard_id in self._shard_events:
                                self._shard_events[shard_id].set()
                    except Exception as e:
                        with self._lock:
                            self._downloading_shards.discard(shard_id)
                            self._error = e
                        logger.error(f"Failed to download shard {shard_id}: {e}")
                        raise  # Re-raise to stop the download thread

        except Exception as e:
            logger.error(f"Download thread error: {e}")
            with self._lock:
                self._error = e

    def _get_shards_to_download(self) -> list:
        """Get the list of shards that should be downloaded to maintain the buffer."""
        with self._lock:
            current = self._current_shard_needed
            end = min(current + self.buffer_size, self.total_shards)

            # Find shards in the buffer range that are not yet downloaded
            shards_to_download = []
            for shard_id in range(current, end):
                if shard_id not in self._downloaded_shards and shard_id not in self._downloading_shards:
                    shards_to_download.append(shard_id)

            return shards_to_download

    def _download_shard(self, shard_id: int):
        """Download a single shard."""
        filename = f"shard_{shard_id:05d}.parquet"
        filepath = os.path.join(self.data_dir, filename)

        # Check if already exists
        if os.path.exists(filepath):
            if self._is_valid_shard(filepath):
                logger.info(f"Skipping {filename} (already exists)")
                return
            else:
                logger.warning(f"Removing corrupted existing file: {filename}")
                os.remove(filepath)

        # Check disk space
        self._check_disk_space()

        # Download
        url = f"{self.base_url}/{filename}"
        logger.info(f"Downloading {filename}...")

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # Write to temporary file first
                temp_path = filepath + ".tmp"
                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                # Move temp file to final location
                os.rename(temp_path, filepath)

                # Verify the downloaded file
                if not self._is_valid_shard(filepath):
                    raise ValueError(f"Downloaded file is not a valid parquet file")

                file_size_mb = downloaded_size / (1024 * 1024)
                with self._lock:
                    num_downloaded = len(self._downloaded_shards) + 1
                logger.info(f"Downloaded {filename} ({file_size_mb:.1f} MB) [{num_downloaded}/{self.total_shards}]")
                return

            except (requests.RequestException, IOError, ValueError) as e:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

                # Clean up any partial files
                for path in [temp_path, filepath]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

                # Retry with exponential backoff
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to download {filename} after {max_attempts} attempts: {e}")

    def _check_disk_space(self):
        """Check available disk space and warn/error if insufficient."""
        try:
            stat = shutil.disk_usage(self.data_dir)
            available_gb = stat.free / (1024 ** 3)

            if available_gb < 1.0:
                raise RuntimeError(f"Insufficient disk space: only {available_gb:.2f} GB available")
            elif available_gb < 5.0:
                logger.warning(f"Low disk space: only {available_gb:.2f} GB available")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

    def wait_for_shard(self, shard_id: int, timeout: float = 300.0) -> bool:
        """
        Wait for a specific shard to be available.

        Args:
            shard_id: The shard ID to wait for
            timeout: Maximum time to wait in seconds (default: 300s = 5 minutes)

        Returns:
            True if the shard is available, False if timeout
        """
        # Update the current shard needed
        with self._lock:
            if shard_id > self._current_shard_needed:
                self._current_shard_needed = shard_id

            # Check if already downloaded
            if shard_id in self._downloaded_shards:
                return True

            # Check for errors in download thread
            if self._error is not None:
                raise RuntimeError(f"Download thread error: {self._error}")

            # Create an event for this shard if it doesn't exist
            if shard_id not in self._shard_events:
                self._shard_events[shard_id] = threading.Event()
            event = self._shard_events[shard_id]

        # Wait for the shard to be downloaded
        logger.info(f"Waiting for shard {shard_id}...")
        start_time = time.time()
        if event.wait(timeout=timeout):
            wait_time = time.time() - start_time
            logger.info(f"Shard {shard_id} ready (waited {wait_time:.1f}s)")
            return True
        else:
            logger.error(f"Timeout waiting for shard {shard_id} after {timeout}s")
            return False

    def stop(self):
        """Stop the download thread cleanly."""
        logger.info("Stopping download thread...")
        self._stop_event.set()

        if self._download_thread is not None:
            self._download_thread.join(timeout=5.0)
            if self._download_thread.is_alive():
                logger.warning("Download thread did not stop cleanly")
            else:
                logger.info("Download thread stopped")
            self._download_thread = None

    def get_status(self) -> dict:
        """
        Get current download status.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "total_shards": self.total_shards,
                "downloaded": len(self._downloaded_shards),
                "downloading": len(self._downloading_shards),
                "current_shard": self._current_shard_needed,
                "buffer_size": self.buffer_size,
                "buffer_available": sum(
                    1 for i in range(self._current_shard_needed,
                                   min(self._current_shard_needed + self.buffer_size, self.total_shards))
                    if i in self._downloaded_shards
                ),
            }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
