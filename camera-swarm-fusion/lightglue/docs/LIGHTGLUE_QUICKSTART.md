# LightGlue Fallback Quick Start Guide

## What is it?
A robust fallback system that recovers ArUco markers that OpenCV fails to detect due to occlusion, motion blur, or poor lighting conditions.

## How does it work?
1. **Template Matching**: Uses SuperPoint+LightGlue to match marker templates against the current frame
2. **Temporal Tracking**: Tracks markers across frames using optical flow when recently seen
3. **Homography Recovery**: Computes corner positions via RANSAC homography estimation

## Quick Setup (3 steps)

### 1. Install Dependencies
```bash
# For CPU (most systems)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lightglue

# For CUDA (if you have NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install lightglue
```

### 2. Generate Templates
```bash
# Create template directory
mkdir -p templates/markers

# Generate templates for your markers
python lightglue/scripts/create_marker_templates.py \
  --marker-ids 0 1 2 3 \
  --dict 4x4_50 \
  --output-dir templates/markers \
  --size 400
```

This creates: `templates/markers/id_0.png`, `id_1.png`, `id_2.png`, `id_3.png`

### 3. Enable in Config
Add the `lightglue` section to your camera config:

```json
{
  "camera_name": "cam1",
  "device": 0,
  "target_ids": [0, 1, 2, 3],
  "lightglue": {
    "enabled": true,
    "device": "cpu",
    "template_dir": "templates/markers"
  }
}
```

## Run It
```bash
python -m camera_fusion.run --config configs/cam1.json
```

## How to Tell It's Working

### In Logs
Look for these messages:
```
INFO - LightGlue fallback initialized
INFO - Loaded 4 templates
INFO - Recovered marker 2 via lightglue_reacquire
```

### In Annotated Frames
Markers are color-coded by detection source:
- **Green markers** = Normal ArUco detection
- **Cyan markers** = Recovered via tracking (fast)
- **Magenta markers** = Recovered via template matching (robust)

Labels show: `ID (LG:track)` or `ID (LG:reacquire)`

## Testing

Test on a saved frame before production use:

```bash
# Capture a test session first (normal ArUco)
python -m camera_fusion.run --config configs/cam1.json --max-frames 20

# Test the fallback on one frame
python lightglue/scripts/test_lightglue_fallback.py \
  --frame data/sessions/cam1_session_*/frames/frame_000010.png \
  --templates templates/markers \
  --marker-ids 0 1 2 3 \
  --device cpu
```

If successful, you'll see:
```
✓ SUCCESS: Recovered at least one marker!
Saved visualization to: .../test_lightglue_result.png
```

## Configuration Tuning

Start with defaults, then adjust if needed:

```json
{
  "lightglue": {
    "enabled": true,
    "device": "cpu",              // or "cuda" for GPU
    "template_dir": "templates/markers",
    "min_inliers": 4,             // Higher = more strict (fewer false positives)
    "max_age_frames": 5,          // How long to track (higher = track longer)
    "roi_expand_px": 50,          // Tracking search area (higher = more robust)
    "debug_save": false,          // Set true to save debug frames
    "corner_refine": true,        // Sub-pixel accuracy (keep true)
    "match_threshold": 0.2        // Lower = stricter matching
  }
}
```

### When to adjust parameters:

**Too many false positives?**
- Increase `min_inliers` (try 6-8)
- Decrease `match_threshold` (try 0.1-0.15)

**Missing valid detections?**
- Decrease `min_inliers` (try 3)
- Increase `match_threshold` (try 0.3-0.4)
- Increase `roi_expand_px` (try 100-150)

**Markers lost between frames?**
- Increase `max_age_frames` (try 10-15)

**Slow performance?**
- Use CUDA device if available
- Decrease `max_age_frames`
- Set `corner_refine` to false (small accuracy loss)

## Performance Expectations

| Operation | CPU Time | CUDA Time |
|-----------|----------|-----------|
| Initialization | 1-3s | 1-3s |
| Tracking (per marker) | 10-30ms | 10-30ms |
| Template match (per marker) | 100-500ms | 20-100ms |

**Example scenario:**
- 4 target markers, 2 detected by ArUco, 2 missing
- Marker 1: Tracked (tracked last frame) → +20ms
- Marker 2: Re-acquired (not seen recently) → +150ms (CPU) or +40ms (CUDA)
- **Total overhead**: ~170ms (CPU) or ~60ms (CUDA)

## Troubleshooting

### "PyTorch not available"
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "LightGlue package not available"
```bash
pip install lightglue
```

### "No templates found"
Check template directory exists and contains `id_<MARKER_ID>.png` files:
```bash
ls -la templates/markers/
```

### "Could not recover any markers"
1. Verify templates match your ArUco dictionary
2. Check template size is reasonable (400px recommended)
3. Ensure markers are actually visible in test frame
4. Try lowering `min_inliers` to 3
5. Enable `debug_save` and inspect debug frames

### Performance too slow
1. Use CUDA if available: `"device": "cuda"`
2. Reduce `max_age_frames`
3. Use smaller templates (200px instead of 400px)

## Advanced: Jetson Platforms

For NVIDIA Jetson devices, use platform-specific PyTorch:

```bash
# Download NVIDIA's pre-built wheel (check version for your JetPack)
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# Install
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# Then install LightGlue
pip install lightglue
```

See `requirements-jetson.txt` for details.

## Example Session Output

With LightGlue enabled, you'll see extended detection data:

```
data/sessions/cam1_session_20260204_120000/
  frames/              # Original frames
  annotated/           # Color-coded markers (green/cyan/magenta)
  debug_lightglue/     # Debug frames (if debug_save=true)
  detections.csv       # All detections (ArUco + recovered)
  logs/session.log     # Recovery events logged
  config.json          # Config snapshot
```

The CSV includes all recovered markers (no special flag needed - they're treated the same as ArUco detections).

## When to Use LightGlue Fallback

**Good use cases:**
- Markers occasionally occluded by robot arms or objects
- Fast-moving cameras (motion blur affects ArUco)
- Variable lighting conditions
- Critical applications where marker loss is unacceptable
- Outdoor environments with shadows

**Not recommended:**
- Static cameras with perfect visibility (overhead)
- Very high-speed capture (>30 fps) - adds latency
- Extremely resource-constrained devices (needs PyTorch)
- Markers never in frame (templates won't help)

## Summary

LightGlue fallback makes your ArUco detection more robust at the cost of some CPU/GPU time. Start with the defaults, test on saved frames, then enable in production with monitoring to see recovery rates in your specific scenario.

For most use cases, the default config works well. Happy tracking! 🎯
