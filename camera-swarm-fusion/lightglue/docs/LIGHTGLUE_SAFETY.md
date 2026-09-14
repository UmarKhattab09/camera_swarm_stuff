# LightGlue Safety Features Documentation

## Overview

The LightGlue fallback system includes comprehensive safety features to ensure robust and reliable marker recovery while preventing false positives and managing computational resources.

## Safety Features

### 1. ID Verification

**Purpose**: Prevent false positives by verifying that recovered corners actually contain the expected ArUco marker ID.

**How it works**:
1. After LightGlue proposes corners for marker ID=X
2. Warp the quadrilateral region to a square patch (200x200 px)
3. Run `cv2.aruco.detectMarkers()` on the warped patch
4. Check if detected ID matches expected ID
5. Reject recovery if mismatch or no decode

**Configuration**:
```json
{
  "lightglue": {
    "verify_id": true  // Enable ID verification (default: true)
  }
}
```

**When to disable**:
- Template quality is very high and false positives are unlikely
- Computational resources are extremely limited
- Running on very low-end hardware

**Performance impact**: ~5-10ms per recovered marker

### 2. Max Fallback Markers Per Frame

**Purpose**: Limit computational overhead by capping the number of fallback attempts per frame.

**How it works**:
1. Sort missing markers by recency (most recently seen first)
2. Attempt recovery for up to `max_fallback_markers_per_frame` markers
3. Skip remaining markers for this frame
4. Prioritization ensures recently-seen markers are recovered first

**Configuration**:
```json
{
  "lightglue": {
    "max_fallback_markers_per_frame": 2  // Max attempts per frame (default: 2)
  }
}
```

**Tuning guidance**:
- **Increase to 3-4**: If you have 5+ target markers and often lose multiple
- **Decrease to 1**: For very low-end hardware or high FPS requirements
- **Keep at 2**: Good balance for most applications

**Example scenario**:
- 4 target markers: [0, 1, 2, 3]
- ArUco detects: [0, 1]
- Missing: [2, 3]
- With max=2: Both will be attempted
- With max=1: Only the most recent will be attempted

### 3. Reacquire Interval Frames

**Purpose**: Prevent expensive template matching from running too frequently on the same marker.

**How it works**:
1. Track last frame index when template reacquire was attempted for each marker
2. Before attempting reacquire, check if enough frames have passed
3. Skip if `current_frame - last_reacquire < reacquire_interval_frames`
4. Tracking (optical flow) is still allowed during the interval

**Configuration**:
```json
{
  "lightglue": {
    "reacquire_interval_frames": 5  // Min frames between reacquire (default: 5)
  }
}
```

**Tuning guidance**:
- **Increase to 10-15**: For slow-moving cameras or static scenes
- **Decrease to 2-3**: For fast camera motion or frequently occluded markers
- **Keep at 5**: Good default for 15 FPS capture

**Example timeline**:
```
Frame 100: Marker 2 reacquired via template matching ✓
Frame 101: Marker 2 missing → tracking attempted (allowed)
Frame 102: Marker 2 missing → tracking failed → reacquire skipped (too soon)
Frame 103: Marker 2 missing → tracking failed → reacquire skipped (too soon)
Frame 104: Marker 2 missing → tracking failed → reacquire skipped (too soon)
Frame 105: Marker 2 missing → tracking failed → reacquire attempted ✓
```

### 4. ROI-Based Matching Preference

**Purpose**: Improve efficiency and reduce false positives by searching only in the region where the marker was last seen.

**How it works**:
1. If `prefer_roi_matching=true` and marker has `last_corners`:
   - Expand ROI around last known position by 3× `roi_expand_px`
   - Extract features only from ROI region
   - Match template against ROI
   - Convert recovered corners back to full frame coordinates
2. If no history or preference disabled:
   - Search entire frame

**Configuration**:
```json
{
  "lightglue": {
    "prefer_roi_matching": true  // Use ROI when possible (default: true)
  }
}
```

**Benefits**:
- **Faster**: Smaller search region = fewer features to extract/match
- **More accurate**: Reduces false matches from similar patterns elsewhere in frame
- **Lower memory**: ROI processing uses less GPU/CPU memory

**When to disable**:
- Markers move very fast (may exit ROI between frames)
- First-time detection (no history available anyway)
- Wide-angle fisheye cameras with distortion

### 5. Output Format Contract

**Purpose**: Ensure compatibility with downstream pose estimation code.

**Guarantees**:
1. **Corner format**: Always `np.ndarray` shape `(4, 2)` dtype `float32`
2. **Coordinate system**: Pixel coordinates in original frame
3. **Corner order**: Matches OpenCV ArUco convention (top-left, top-right, bottom-right, bottom-left)
4. **No invalid values**: All corners within frame bounds
5. **Source annotation**: Every detection includes source in DebugInfo

**Validation checks**:
```python
# Automatic validation in recover_missing()
if recovered_corners.shape != (4, 2):
    logger.warning("Invalid corner shape")
    continue
if recovered_corners.dtype != np.float32:
    recovered_corners = recovered_corners.astype(np.float32)
if (corners < 0).any() or (corners >= frame_size).any():
    logger.warning("Corners out of bounds")
    return None
```

**Source annotations**:
- `"aruco"`: Detected by OpenCV ArUco
- `"lg_track"`: Recovered via optical flow tracking
- `"lg_reacquire"`: Recovered via template matching

## Configuration Reference

Complete safety configuration:

```json
{
  "lightglue": {
    "enabled": true,
    "device": "cpu",
    "template_dir": "templates/markers",
    
    // Core parameters
    "min_inliers": 4,
    "max_age_frames": 5,
    "roi_expand_px": 50,
    "corner_refine": true,
    "match_threshold": 0.2,
    "debug_save": false,
    
    // Safety features
    "verify_id": true,                      // NEW: ID verification
    "max_fallback_markers_per_frame": 2,    // NEW: Limit attempts per frame
    "reacquire_interval_frames": 5,         // NEW: Min frames between reacquire
    "prefer_roi_matching": true             // NEW: Use ROI when possible
  }
}
```

## Validation Workflow

Use the validation script to test on saved sessions:

```bash
# Basic validation (with fallback)
python lightglue/scripts/validate_lightglue_on_session.py \
  --session data/sessions/cam1_session_20260204_120000 \
  --templates templates/markers

# Baseline comparison (no fallback)
python lightglue/scripts/validate_lightglue_on_session.py \
  --session data/sessions/cam1_session_20260204_120000 \
  --no-fallback

# With debug overlays
python lightglue/scripts/validate_lightglue_on_session.py \
  --session data/sessions/cam1_session_20260204_120000 \
  --templates templates/markers \
  --output debug_overlays/

# Test without ID verification
python lightglue/scripts/validate_lightglue_on_session.py \
  --session data/sessions/cam1_session_20260204_120000 \
  --templates templates/markers \
  --no-verify-id
```

## Validation Output

The script generates comprehensive statistics:

```
VALIDATION RESULTS
======================================================================
Total frames processed: 150
Frames with all expected markers: 142
Frames where fallback was used: 28

Per-Marker Detection Statistics:
----------------------------------------------------------------------
Marker ID    ArUco      LG Track    LG Reacq    Total      Rate    
----------------------------------------------------------------------
0            148        2           0           150        100.0%
1            145        3           2           150        100.0%
2            138        8           4           150        100.0%
3            142        6           2           150        100.0%
----------------------------------------------------------------------

Fallback Impact:
  Frames where fallback helped: 28
  Total recoveries: 27
    Via tracking: 19
    Via reacquire: 8
```

**Key metrics**:
- **Detection rate**: Percentage of frames where each marker was detected
- **ArUco count**: Detections by standard ArUco (baseline)
- **LG Track**: Recoveries via optical flow (fast, <30ms)
- **LG Reacq**: Recoveries via template matching (robust, 100-500ms)

## Performance Impact

With all safety features enabled:

| Scenario | Overhead (CPU) | Overhead (CUDA) |
|----------|----------------|-----------------|
| Tracking only (1 marker) | 15-35ms | 15-35ms |
| Reacquire only (1 marker) | 110-520ms | 25-110ms |
| ID verification (per marker) | 5-10ms | 5-10ms |
| ROI matching vs full frame | -60% time | -60% time |

**Example frame processing**:
- 2 markers missing
- 1 recovered via tracking: 20ms + 7ms (verify) = 27ms
- 1 recovered via reacquire (ROI): 80ms + 7ms (verify) = 87ms
- Total overhead: ~114ms
- Suitable for 10 FPS applications

## Best Practices

### 1. Start with Defaults
The default values are tuned for general use:
```json
{
  "verify_id": true,
  "max_fallback_markers_per_frame": 2,
  "reacquire_interval_frames": 5,
  "prefer_roi_matching": true
}
```

### 2. Monitor Validation Stats
Run validation script on representative sessions to understand:
- How often fallback is needed
- Which markers are problematic
- Tracking vs reacquire ratio

### 3. Tune Based on Use Case

**High FPS (>20 FPS), low latency**:
```json
{
  "max_fallback_markers_per_frame": 1,
  "reacquire_interval_frames": 10,
  "verify_id": false
}
```

**Maximum robustness**:
```json
{
  "max_fallback_markers_per_frame": 4,
  "reacquire_interval_frames": 2,
  "verify_id": true,
  "min_inliers": 6
}
```

**Power-constrained device**:
```json
{
  "max_fallback_markers_per_frame": 1,
  "reacquire_interval_frames": 15,
  "prefer_roi_matching": true,
  "roi_expand_px": 30
}
```

### 4. Check Debug Logs

Enable debug logging to see safety features in action:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Look for messages:
- "ID verification passed/failed"
- "Reached max fallback attempts"
- "Skipping reacquire: only X frames since last attempt"
- "Using ROI search for marker X"

## Troubleshooting

### Too many false positives
✅ **Solution**: 
- Increase `min_inliers` (try 6-8)
- Ensure `verify_id=true`
- Decrease `match_threshold` (try 0.1-0.15)

### Missing valid detections
✅ **Solution**:
- Increase `max_fallback_markers_per_frame`
- Decrease `reacquire_interval_frames`
- Disable `verify_id` if templates are perfect matches

### Performance too slow
✅ **Solution**:
- Use CUDA device
- Decrease `max_fallback_markers_per_frame`
- Increase `reacquire_interval_frames`
- Ensure `prefer_roi_matching=true`

### ID verification always fails
✅ **Solution**:
- Check template quality (should be clean ArUco markers)
- Regenerate templates with `create_marker_templates.py`
- Verify ArUco dictionary matches (`aruco_dict`)
- Consider disabling verification if templates differ from actual markers

## Safety Feature Summary

All safety features work together to provide:

✅ **Accuracy**: ID verification prevents false positives  
✅ **Efficiency**: ROI matching and reacquire intervals reduce computation  
✅ **Robustness**: Max attempts ensures frame processing doesn't stall  
✅ **Compatibility**: Output format contract ensures downstream code works  
✅ **Debuggability**: Source annotations track recovery methods  

The system gracefully handles edge cases and provides clear logging for troubleshooting.
