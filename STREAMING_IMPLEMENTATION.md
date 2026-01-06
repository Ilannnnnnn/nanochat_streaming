# Progressive Shard Download - Step 1 Implementation

## Summary

This implementation adds progressive shard downloading to nanochat, allowing training to proceed while downloading shards on-demand. This saves disk space by maintaining only a buffer of shards ahead of the current training position.

## Changes Made

### 1. New Module: `nanochat/streaming_downloader.py`

Created a new `StreamingDownloader` class that:
- Downloads shards progressively in a background thread
- Maintains a configurable buffer of N shards ahead (default: 5)
- Thread-safe operations using locks and events
- Blocks training if needed shard is not yet available
- Validates downloaded shards to detect corruption
- Provides status information about download progress

**Key Features:**
- Initial buffer download before training starts
- Automatic retry with exponential backoff for failed downloads
- Disk space checking (warns at <5GB, errors at <1GB)
- Detects and removes corrupted parquet files
- Clean thread shutdown on completion

### 2. Modified: `nanochat/dataloader.py`

Added support for streaming downloader:
- New parameter `streaming_downloader` to `tokenizing_distributed_data_loader_with_state()`
- Helper function `extract_shard_id()` to parse shard IDs from filenames
- Before reading each parquet file, waits for shard availability if streaming is enabled
- Logs when starting to read each shard

### 3. Modified: `scripts/base_train.py`

Added streaming mode configuration:
- New config parameters:
  - `streaming_mode` (bool): Enable/disable streaming (default: False)
  - `streaming_buffer_size` (int): Number of shards to buffer (default: 5)
  - `streaming_total_shards` (int): Total shards to download (default: -1 = auto-calculate)
- Auto-calculates required shards based on total tokens needed
- Creates and starts StreamingDownloader when streaming_mode=True
- Logs streaming status every 100 steps
- Clean shutdown of downloader on training completion

### 4. Tests: `tests/test_streaming_downloader.py`

Comprehensive unit tests covering:
- Shard ID extraction from filenames
- StreamingDownloader initialization
- Scanning for existing shards
- Individual shard download
- Wait/timeout behavior
- Status reporting
- Buffer management logic
- Thread cleanup
- Context manager usage

## Usage Instructions

### Basic Streaming Mode

To enable streaming mode with default settings:

```bash
python -m scripts.base_train \
    --depth=4 \
    --device_batch_size=1 \
    --num_iterations=100 \
    --streaming_mode=True
```

### Custom Buffer Size

To use a custom buffer size (e.g., 10 shards):

```bash
python -m scripts.base_train \
    --depth=4 \
    --device_batch_size=1 \
    --num_iterations=100 \
    --streaming_mode=True \
    --streaming_buffer_size=10
```

### Explicit Shard Count

To explicitly set the number of shards to download:

```bash
python -m scripts.base_train \
    --depth=4 \
    --device_batch_size=1 \
    --num_iterations=100 \
    --streaming_mode=True \
    --streaming_buffer_size=5 \
    --streaming_total_shards=20
```

### Minimal Test (Recommended for First Test)

Quick test with minimal resources:

```bash
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=50 \
    --streaming_mode=True \
    --streaming_buffer_size=3 \
    --streaming_total_shards=5 \
    --eval_tokens=512 \
    --core_metric_every=-1
```

## Expected Behavior

### Startup
```
Streaming mode: calculated 10 shards needed for 5,242,880 tokens
[StreamingDownloader] Initialized StreamingDownloader: 10 shards, buffer_size=5
[StreamingDownloader] Data directory: /home/user/.cache/nanochat/base_data
[StreamingDownloader] Found 0 shards already downloaded
[StreamingDownloader] Starting download of 10 shards with buffer_size=5
[StreamingDownloader] Downloading initial buffer: shards 0-4
[StreamingDownloader] Downloading shard_00000.parquet...
[StreamingDownloader] Downloaded shard_00000.parquet (250.3 MB) [1/10]
...
[StreamingDownloader] Initial buffer ready, training can start
```

### During Training
```
step 00050/00100 (50.00%) | loss: 9.234567 | lrm: 1.00 | dt: 245.12ms | tok/sec: 125,000 | mfu: 45.23 | total time: 0.20m
[Streaming] 7/10 shards downloaded, buffer: 5/5, current: shard_00001
[Dataloader] Reading shard_00002.parquet
[StreamingDownloader] Downloading shard_00007.parquet...
```

### Completion
```
Stopping streaming downloader...
[StreamingDownloader] Stopping download thread...
[StreamingDownloader] Download thread stopped
```

## Testing

### Unit Tests

Run the unit tests (requires pytest or unittest):

```bash
# With pytest (if installed)
pytest tests/test_streaming_downloader.py -v

# With unittest
python -m unittest tests.test_streaming_downloader -v
```

### Integration Test

Monitor disk usage in real-time while training:

```bash
# Terminal 1: Run training
python -m scripts.base_train \
    --depth=4 \
    --device_batch_size=1 \
    --num_iterations=100 \
    --streaming_mode=True \
    --streaming_buffer_size=3

# Terminal 2: Monitor disk usage
watch -n 1 'ls -lh ~/.cache/nanochat/base_data/ | tail -20'
```

You should observe:
- Shards appearing progressively in the data directory
- Buffer of N shards maintained ahead of current position
- Training proceeding without blocking (if buffer is sufficient)

### Validation Checklist

- [ ] Training starts without errors
- [ ] Initial buffer downloads before training begins
- [ ] Shards appear progressively in data directory
- [ ] Streaming status logs every 100 steps
- [ ] Training doesn't block waiting for shards (buffer maintained)
- [ ] Download thread stops cleanly on completion
- [ ] Already downloaded shards are reused on restart

## Performance Characteristics

With your system specs (800 Mbit/s = 100 MB/s):
- Download time per shard (~250MB): ~2.5 seconds
- Buffer of 5 shards = ~1.25GB disk usage
- If each shard takes >30 seconds to consume, no blocking expected

## Known Limitations (Step 1)

This is Step 1 of the implementation, with intentional limitations:

1. **No automatic shard deletion**: Downloaded shards remain on disk
2. **Basic error handling**: Network errors cause training to crash
3. **No retry logic**: Failed downloads fail immediately (after 5 attempts)
4. **No state persistence**: Crash recovery starts from shard 0
5. **Fixed buffer size**: No dynamic adaptation based on download/consumption rate

These will be addressed in subsequent implementation steps.

## Troubleshooting

### Training blocks waiting for shard

**Cause**: Buffer too small or network too slow
**Solution**: Increase `streaming_buffer_size` or check network connection

```bash
--streaming_buffer_size=10  # Increase from default 5
```

### Disk space warnings

**Cause**: Less than 5GB available
**Solution**: Free up disk space or reduce buffer size

```bash
--streaming_buffer_size=2  # Reduce buffer
```

### Download failures

**Cause**: Network issues or HuggingFace unavailable
**Solution**: Check network connection, retry later, or download manually:

```bash
python -m nanochat.dataset -n 20 -w 4
```

### Import errors in tests

**Cause**: Missing dependencies (torch, pyarrow, etc.)
**Solution**: Install requirements:

```bash
pip install torch pyarrow requests
```

## Files Modified

1. **Created**: `nanochat/streaming_downloader.py` (385 lines)
2. **Modified**: `nanochat/dataloader.py` (added streaming support)
3. **Modified**: `scripts/base_train.py` (added streaming mode)
4. **Created**: `tests/test_streaming_downloader.py` (384 lines)

## Next Steps (Future Iterations)

Step 2 will add:
- Automatic deletion of consumed shards
- Configurable retention policy (keep N shards behind current position)
- Disk space management

Step 3 will add:
- Robust retry logic for download failures
- Network error recovery
- Graceful degradation on persistent failures

Step 4 will add:
- State persistence in checkpoints
- Resume from exact shard position after crash
- Validation of state consistency

Step 5 will add:
- Dynamic buffer size adjustment
- Multi-GPU coordination
- Performance optimizations

## Questions & Support

For issues or questions:
1. Check logs for error messages
2. Verify disk space with `df -h`
3. Test network connectivity to HuggingFace
4. Review this document's troubleshooting section
