"""
Unit tests for the streaming downloader module.

These tests verify the core functionality of progressive shard downloading,
buffer management, and thread-safe operations.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import threading

from nanochat.streaming_downloader import StreamingDownloader
from nanochat.dataloader import extract_shard_id


class TestExtractShardId(unittest.TestCase):
    """Test the shard ID extraction helper function."""

    def test_extract_shard_id_basic(self):
        """Test basic shard ID extraction."""
        filepath = "/path/to/shard_00042.parquet"
        self.assertEqual(extract_shard_id(filepath), 42)

    def test_extract_shard_id_zero(self):
        """Test extraction of shard 0."""
        filepath = "shard_00000.parquet"
        self.assertEqual(extract_shard_id(filepath), 0)

    def test_extract_shard_id_large(self):
        """Test extraction of large shard ID."""
        filepath = "/some/dir/shard_01822.parquet"
        self.assertEqual(extract_shard_id(filepath), 1822)

    def test_extract_shard_id_invalid(self):
        """Test that invalid filenames raise ValueError."""
        with self.assertRaises(ValueError):
            extract_shard_id("invalid_file.parquet")


class TestStreamingDownloader(unittest.TestCase):
    """Test the StreamingDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test StreamingDownloader initialization."""
        downloader = StreamingDownloader(
            total_shards=10,
            buffer_size=3,
            data_dir=self.temp_dir,
            num_workers=2
        )

        self.assertEqual(downloader.total_shards, 10)
        self.assertEqual(downloader.buffer_size, 3)
        self.assertEqual(downloader.num_workers, 2)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_scan_existing_shards(self):
        """Test scanning for already downloaded shards."""
        # Create some fake shard files
        for i in range(3):
            filepath = os.path.join(self.temp_dir, f"shard_{i:05d}.parquet")
            # Create a minimal parquet file (we'll mock validation)
            with open(filepath, 'wb') as f:
                f.write(b'P' * (2 * 1024 * 1024))  # 2MB fake file

        # Mock the validation to return True
        with patch.object(StreamingDownloader, '_is_valid_shard', return_value=True):
            downloader = StreamingDownloader(
                total_shards=10,
                buffer_size=3,
                data_dir=self.temp_dir
            )

            # Should have found 3 shards
            self.assertEqual(len(downloader._downloaded_shards), 3)
            self.assertIn(0, downloader._downloaded_shards)
            self.assertIn(1, downloader._downloaded_shards)
            self.assertIn(2, downloader._downloaded_shards)

    @patch('nanochat.streaming_downloader.requests.get')
    @patch('pyarrow.parquet.ParquetFile')
    def test_download_shard(self, mock_pf, mock_get):
        """Test downloading a single shard."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b'test_data' * 1000]
        mock_get.return_value = mock_response

        # Mock parquet validation
        mock_pf.return_value = MagicMock()

        downloader = StreamingDownloader(
            total_shards=5,
            buffer_size=2,
            data_dir=self.temp_dir
        )

        # Download a shard
        downloader._download_shard(0)

        # Verify file was created
        filepath = os.path.join(self.temp_dir, "shard_00000.parquet")
        self.assertTrue(os.path.exists(filepath))

        # Verify HTTP request was made
        mock_get.assert_called_once()

    def test_wait_for_shard_already_downloaded(self):
        """Test waiting for a shard that's already downloaded."""
        downloader = StreamingDownloader(
            total_shards=5,
            buffer_size=2,
            data_dir=self.temp_dir
        )

        # Mark shard as downloaded
        downloader._downloaded_shards.add(0)

        # Should return immediately
        start_time = time.time()
        result = downloader.wait_for_shard(0, timeout=5.0)
        elapsed = time.time() - start_time

        self.assertTrue(result)
        self.assertLess(elapsed, 1.0)  # Should be nearly instant

    def test_wait_for_shard_timeout(self):
        """Test waiting for a shard that never arrives."""
        downloader = StreamingDownloader(
            total_shards=5,
            buffer_size=2,
            data_dir=self.temp_dir
        )

        # Don't download anything, just wait
        start_time = time.time()
        result = downloader.wait_for_shard(0, timeout=2.0)
        elapsed = time.time() - start_time

        self.assertFalse(result)  # Should timeout
        self.assertGreaterEqual(elapsed, 2.0)  # Should take at least 2 seconds

    def test_wait_for_shard_signals_when_ready(self):
        """Test that wait_for_shard unblocks when shard becomes available."""
        downloader = StreamingDownloader(
            total_shards=5,
            buffer_size=2,
            data_dir=self.temp_dir
        )

        result_holder = []

        def wait_thread():
            result = downloader.wait_for_shard(0, timeout=5.0)
            result_holder.append(result)

        # Start waiting in a separate thread
        thread = threading.Thread(target=wait_thread)
        thread.start()

        # Wait a bit, then signal the shard is ready
        time.sleep(0.5)
        with downloader._lock:
            downloader._downloaded_shards.add(0)
            if 0 in downloader._shard_events:
                downloader._shard_events[0].set()

        # Wait for thread to finish
        thread.join(timeout=2.0)

        # Should have succeeded
        self.assertEqual(len(result_holder), 1)
        self.assertTrue(result_holder[0])

    def test_get_status(self):
        """Test getting downloader status."""
        downloader = StreamingDownloader(
            total_shards=10,
            buffer_size=5,
            data_dir=self.temp_dir
        )

        # Mark some shards as downloaded
        downloader._downloaded_shards.add(0)
        downloader._downloaded_shards.add(1)
        downloader._downloaded_shards.add(2)

        # Mark some as downloading
        downloader._downloading_shards.add(3)

        # Set current shard
        downloader._current_shard_needed = 1

        status = downloader.get_status()

        self.assertEqual(status['total_shards'], 10)
        self.assertEqual(status['downloaded'], 3)
        self.assertEqual(status['downloading'], 1)
        self.assertEqual(status['current_shard'], 1)
        self.assertEqual(status['buffer_size'], 5)

    def test_get_shards_to_download(self):
        """Test determining which shards to download for the buffer."""
        downloader = StreamingDownloader(
            total_shards=10,
            buffer_size=5,
            data_dir=self.temp_dir
        )

        # Current shard is 0, buffer is 5
        # Should want to download shards 0-4
        downloader._current_shard_needed = 0
        shards = downloader._get_shards_to_download()
        self.assertEqual(shards, [0, 1, 2, 3, 4])

        # Mark some as downloaded
        downloader._downloaded_shards.add(0)
        downloader._downloaded_shards.add(1)
        shards = downloader._get_shards_to_download()
        self.assertEqual(shards, [2, 3, 4])

        # Mark some as downloading
        downloader._downloading_shards.add(2)
        shards = downloader._get_shards_to_download()
        self.assertEqual(shards, [3, 4])

        # Move to next shard
        downloader._current_shard_needed = 1
        shards = downloader._get_shards_to_download()
        # Should now want shards 1-5, but 1 is downloaded and 2 is downloading
        self.assertEqual(shards, [3, 4, 5])

    def test_stop(self):
        """Test stopping the download thread cleanly."""
        downloader = StreamingDownloader(
            total_shards=5,
            buffer_size=2,
            data_dir=self.temp_dir
        )

        # Start with mocked download to avoid actual HTTP requests
        with patch.object(downloader, '_download_shard'):
            downloader._download_thread = threading.Thread(target=downloader._download_loop, daemon=True)
            downloader._download_thread.start()

            # Give it a moment to start
            time.sleep(0.1)

            # Stop it
            downloader.stop()

            # Thread should be stopped
            self.assertFalse(downloader._download_thread.is_alive())

    def test_context_manager(self):
        """Test using StreamingDownloader as a context manager."""
        with patch.object(StreamingDownloader, 'start') as mock_start, \
             patch.object(StreamingDownloader, 'stop') as mock_stop, \
             patch.object(StreamingDownloader, '_wait_for_initial_buffer'):

            with StreamingDownloader(total_shards=5, buffer_size=2, data_dir=self.temp_dir) as downloader:
                self.assertIsInstance(downloader, StreamingDownloader)

            # Should have called start and stop
            mock_start.assert_called_once()
            mock_stop.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('nanochat.streaming_downloader.requests.get')
    @patch('pyarrow.parquet.ParquetFile')
    def test_buffer_maintenance(self, mock_pf, mock_get):
        """Test that the buffer is maintained correctly during downloads."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b'test_data' * 1000]
        mock_get.return_value = mock_response

        # Mock parquet validation
        mock_pf.return_value = MagicMock()

        downloader = StreamingDownloader(
            total_shards=10,
            buffer_size=3,
            data_dir=self.temp_dir
        )

        # Start the downloader (mocked to avoid long waits)
        with patch.object(downloader, '_wait_for_initial_buffer'):
            downloader.start()

        # Give it a moment to download
        time.sleep(1.0)

        # Check status - should have downloaded initial buffer
        status = downloader.get_status()
        self.assertGreaterEqual(status['downloaded'], 1)

        # Stop cleanly
        downloader.stop()


if __name__ == '__main__':
    unittest.main()
