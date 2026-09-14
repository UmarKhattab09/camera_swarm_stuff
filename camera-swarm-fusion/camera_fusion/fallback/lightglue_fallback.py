"""LightGlue-based fallback for ArUco marker detection.

When OpenCV ArUco fails to detect expected markers, this module uses
SuperPoint+LightGlue feature matching with homography-based corner recovery.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TrackerState:
    """Tracks last known state of a marker for temporal coherence."""
    marker_id: int
    last_corners: np.ndarray  # (4, 2) float32
    last_seen_frame_index: int
    last_frame_gray: Optional[np.ndarray] = None


@dataclass
class DebugInfo:
    """Debug information for recovered markers."""
    marker_id: int
    source: str  # "aruco", "lightglue_reacquire", "lightglue_track"
    inliers: int
    match_quality: float
    homography: Optional[np.ndarray] = None
    raw_match_count: int = 0
    ransac_inlier_count: int = 0
    inlier_ratio: float = 0.0
    superpoint_ms: float = 0.0
    lightglue_ms: float = 0.0
    ransac_ms: float = 0.0
    fallback_total_ms: float = 0.0
    success: bool = True
    failure_reason: Optional[str] = None


class LightGlueFallback:
    """LightGlue-based fallback system for recovering missing ArUco markers.
    
    Uses SuperPoint feature detection and LightGlue matching to:
    1. Track markers across frames using optical flow
    2. Re-acquire markers by matching against templates
    3. Compute homographies to recover corner positions
    """

    def __init__(self, lightglue_cfg, aruco_cfg):
        """Initialize LightGlue fallback system.
        
        Args:
            lightglue_cfg: LightGlueConfig with settings
            aruco_cfg: ArUco configuration for marker info
        """
        self.cfg = lightglue_cfg
        self.aruco_cfg = aruco_cfg
        self.device = lightglue_cfg.device
        self.enabled = lightglue_cfg.enabled
        self.debug_matches_dir = Path(lightglue_cfg.debug_matches_dir) if getattr(lightglue_cfg, "debug_matches_dir", None) else None
        
        # Tracker state per marker ID
        self.tracker_states: Dict[int, TrackerState] = {}
        self.current_frame_index = 0
        
        # Reacquire interval tracking: marker_id -> last_reacquire_frame_index
        self.last_reacquire_frame: Dict[int, int] = {}
        
        # ArUco detector for ID verification
        self.aruco_detector = None
        
        # Template data: marker_id -> dict with image, keypoints (torch), descriptors (torch), corners
        self.templates: Dict[int, Dict] = {}
        
        # SuperPoint + LightGlue models
        self.superpoint = None
        self.lightglue = None
        self.torch_available = False
        self._last_frame_stats: dict[str, Any] = {
            "fallback_triggered": False,
            "fallback_success": False,
            "raw_match_count": 0,
            "ransac_inlier_count": 0,
            "inlier_ratio": 0.0,
            "superpoint_ms": 0.0,
            "lightglue_ms": 0.0,
            "ransac_ms": 0.0,
            "fallback_total_ms": 0.0,
            "failure_reason": None,
        }
        
        if not self.enabled:
            logger.info("LightGlue fallback is disabled")
            return
        
        # Log configuration
        logger.info(f"[LightGlue] Initializing LightGlue fallback with template_dir: {Path(self.cfg.template_dir).resolve()}")
        logger.info(f"[LightGlue] Debug save: {self.cfg.debug_save}, device: {self.device}")
        
        # Try to import and initialize torch/lightglue
        try:
            import torch
            self.torch_available = True
            logger.info(f"PyTorch available, using device: {self.device}")
            
            # Import SuperPoint and LightGlue
            try:
                from lightglue import SuperPoint, LightGlue
                from lightglue.utils import load_image, rbd
                
                self.SuperPoint = SuperPoint
                self.LightGlue = LightGlue
                self.load_image = load_image
                self.rbd = rbd
                
                # Initialize models
                self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
                self.lightglue = LightGlue(features='superpoint').eval().to(self.device)
                
                logger.info("SuperPoint and LightGlue models initialized successfully")
                
                # Initialize ArUco detector for ID verification
                if self.cfg.verify_id:
                    self._init_aruco_detector()
                    
            except ImportError as e:
                logger.warning(f"LightGlue package not available: {e}")
                logger.warning("Install with: pip install lightglue")
                self.enabled = False
                return
                
        except ImportError:
            logger.warning("PyTorch not available - LightGlue fallback disabled")
            logger.warning("Install with: pip install torch")
            self.enabled = False
            return
        
        # Load templates
        self._load_templates()

    def _save_correspondence_debug_image(
        self,
        marker_id: int,
        template_image: np.ndarray,
        frame_image: np.ndarray,
        template_kpts,
        frame_kpts,
        match_pairs,
        frame_index: int,
        label: str,
        raw_match_count: int = 0,
    ) -> None:
        """Save correspondence debug image even if matching failed.
        
        Args:
            marker_id: Marker ID
            template_image: Template image (numpy array)
            frame_image: Frame/ROI image (numpy array)
            template_kpts: Template keypoints (torch tensor or numpy array)
            frame_kpts: Frame keypoints (torch tensor or numpy array)
            match_pairs: Matched keypoint indices (torch tensor or numpy array, can be empty)
            frame_index: Current frame index
            label: Label for the debug image
            raw_match_count: Number of raw matches (for logging)
        """
        if not self.cfg.debug_save or self.debug_matches_dir is None:
            return

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from lightglue.viz2d import plot_images, plot_matches
        except Exception as exc:
            logger.debug("LightGlue correspondence visualization unavailable: %s", exc)
            return

        try:
            # Ensure directory exists
            self.debug_matches_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = self.debug_matches_dir / (
                f"frame_{frame_index:06d}_marker_{marker_id}_{label}_{time.time_ns()}.png"
            )

            plt.close("all")
            
            # Determine title based on match count
            if raw_match_count == 0:
                title_suffix = " (0 matches)"
            else:
                title_suffix = f" ({raw_match_count} matches)"
            
            plot_images([template_image, frame_image], titles=[f"template{title_suffix}", "frame/roi"])

            # Plot matches only if we have any
            if match_pairs is not None and len(match_pairs) > 0:
                if hasattr(template_kpts, "detach"):
                    template_kpts = template_kpts.detach().cpu()
                if hasattr(frame_kpts, "detach"):
                    frame_kpts = frame_kpts.detach().cpu()
                if hasattr(match_pairs, "detach"):
                    match_pairs = match_pairs.detach().cpu()

                m_kpts0 = template_kpts[match_pairs[:, 0]]
                m_kpts1 = frame_kpts[match_pairs[:, 1]]
                try:
                    plot_matches(m_kpts0, m_kpts1)
                except Exception as e:
                    logger.debug("Failed to plot match lines: %s", e)

            plt.savefig(str(out_path))
            plt.close(plt.gcf())
            logger.info("Saved LightGlue correspondence image: %s", out_path)
        except Exception as exc:
            logger.debug(
                "Failed to save LightGlue correspondence image for marker %s: %s",
                marker_id,
                exc,
            )
    
    def _init_aruco_detector(self):
        """Initialize ArUco detector for ID verification."""
        try:
            from camera_fusion.detect import build_detector
            _, _, self.aruco_detector = build_detector(self.aruco_cfg)
            logger.info("ArUco detector initialized for ID verification")
        except Exception as e:
            logger.warning(f"Failed to initialize ArUco detector for verification: {e}")
            self.aruco_detector = None
    
    def _verify_marker_id(
        self,
        frame_gray: np.ndarray,
        corners: np.ndarray,
        expected_id: int
    ) -> bool:
        """Verify that the warped patch at corners contains the expected ArUco ID.
        
        Args:
            frame_gray: Frame in grayscale
            corners: (4, 2) corner coordinates
            expected_id: Expected marker ID
        
        Returns:
            True if ID matches, False otherwise
        """
        if not self.cfg.verify_id or self.aruco_detector is None:
            return True  # Skip verification if disabled or detector unavailable
        
        try:
            # Define destination square (200x200 px)
            patch_size = 200
            dst_corners = np.array([
                [0, 0],
                [patch_size - 1, 0],
                [patch_size - 1, patch_size - 1],
                [0, patch_size - 1]
            ], dtype=np.float32)
            
            # Compute homography to warp quadrilateral to square
            H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
            
            # Warp the patch
            warped = cv2.warpPerspective(frame_gray, H, (patch_size, patch_size))
            
            # Run ArUco detection on warped patch
            if self.aruco_detector is not None:
                corners_detected, ids, _ = self.aruco_detector.detectMarkers(warped)
            else:
                # Fallback to legacy API
                from camera_fusion.detect import build_detector
                dictionary, params, _ = build_detector(self.aruco_cfg)
                corners_detected, ids, _ = cv2.aruco.detectMarkers(warped, dictionary, parameters=params)
            
            # Check if expected ID was detected
            if ids is not None and len(ids) > 0:
                detected_ids = ids.flatten()
                if expected_id in detected_ids:
                    logger.debug(f"ID verification passed for marker {expected_id}")
                    return True
                else:
                    logger.warning(f"ID verification failed: expected {expected_id}, got {detected_ids}")
                    return False
            else:
                logger.warning(f"ID verification failed: no marker detected in warped patch for ID {expected_id}")
                return False
                
        except Exception as e:
            logger.debug(f"ID verification failed with exception for marker {expected_id}: {e}")
            return False
    
    def _load_templates(self):
        """Load marker templates and precompute their features.
        
        Template features (keypoints & descriptors) are computed ONCE and cached
        as torch tensors on the configured device. This avoids recomputing them
        per frame during reacquire attempts.
        """
        if not self.enabled:
            return
        
        template_dir = Path(self.cfg.template_dir)
        logger.info(f"[LightGlue] Starting template loading from: {template_dir.resolve()}")
        
        if not template_dir.exists():
            logger.error(f"[LightGlue] Template directory not found: {template_dir.resolve()}")
            logger.error("[LightGlue] LightGlue fallback will not be able to recover markers")
            return
        
        import torch
        
        # Find all template files
        template_files = sorted(template_dir.glob("id_*.png"))
        logger.info(f"[LightGlue] Found {len(template_files)} template files in {template_dir.name}")
        
        if not template_files:
            logger.warning(f"[LightGlue] No template files (id_*.png) found in {template_dir.resolve()}")
            return
        
        # Look for templates named id_<ID>.png
        for template_path in template_files:
            try:
                marker_id = int(template_path.stem.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"[LightGlue] Invalid template filename: {template_path.name}")
                continue
            
            # Load template image
            img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"[LightGlue] Failed to load template image: {template_path.resolve()}")
                continue
            
            logger.info(f"[LightGlue] Loading template for marker_id={marker_id} from {template_path.resolve()}")
            
            # Extract features using SuperPoint (ONCE at init time)
            h, w = img.shape
            logger.debug(f"[LightGlue] Template image size: {w}x{h}")
            
            # Convert to torch tensor and normalize
            img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).to(self.device)[None, None]
            
            with torch.no_grad():
                feats = self.superpoint({'image': img_tensor})
            
            # Keep keypoints and descriptors as torch tensors on device
            # feats['keypoints'] is shape (1, N, 2)
            # feats['descriptors'] is shape (1, 256, N)
            template_kpts = feats['keypoints'][0]  # (N, 2) on device
            template_desc = feats['descriptors'][0].T  # (N, 256) on device
            
            # Define template corners (assuming template is the marker square)
            corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)
            
            # Store template data with torch tensors
            self.templates[marker_id] = {
                'image': img,
                'keypoints': template_kpts,  # torch.Tensor (N, 2) on device
                'descriptors': template_desc,  # torch.Tensor (N, 256) on device
                'corners': corners
            }
            logger.info(f"[LightGlue] ✓ Loaded template for marker_id={marker_id}: {int(template_kpts.shape[0])} keypoints from SuperPoint")
        
        logger.info(f"[LightGlue] ✓ Template loading complete: {len(self.templates)} marker templates loaded")
        if self.templates:
            loaded_ids = sorted(self.templates.keys())
            logger.info(f"[LightGlue] Loaded template IDs: {loaded_ids}")
    
    def recover_missing(
        self,
        frame_bgr: np.ndarray,
        detected_dict: Dict[int, np.ndarray],
        expected_ids: Set[int]
    ) -> Tuple[Dict[int, np.ndarray], List[DebugInfo]]:
        """Recover missing markers using LightGlue.
        
        Args:
            frame_bgr: Current frame in BGR format
            detected_dict: Dict mapping marker_id -> corners (4, 2) for detected markers
            expected_ids: Set of marker IDs we expect to see
        
        Returns:
            (updated_detected_dict, debug_info_list)
        """
        if not self.enabled:
            self._last_frame_stats = {
                "fallback_triggered": False,
                "fallback_success": False,
                "raw_match_count": 0,
                "ransac_inlier_count": 0,
                "inlier_ratio": 0.0,
                "superpoint_ms": 0.0,
                "lightglue_ms": 0.0,
                "ransac_ms": 0.0,
                "fallback_total_ms": 0.0,
                "failure_reason": None,
            }
            return detected_dict, []
        
        debug_info: List[DebugInfo] = []
        updated_dict = detected_dict.copy()
        
        # Increment frame counter
        self.current_frame_index += 1
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Update tracker states for detected markers (with proper source annotation)
        for marker_id, corners in detected_dict.items():
            self.tracker_states[marker_id] = TrackerState(
                marker_id=marker_id,
                last_corners=corners.copy(),
                last_seen_frame_index=self.current_frame_index,
                last_frame_gray=frame_gray.copy()
            )
            debug_info.append(DebugInfo(
                marker_id=marker_id,
                source="aruco",
                inliers=0,
                match_quality=1.0
            ))
        
        # Identify missing markers
        detected_ids = set(detected_dict.keys())
        missing_ids = expected_ids - detected_ids
        
        if not missing_ids:
            self._last_frame_stats = {
                "fallback_triggered": False,
                "fallback_success": False,
                "raw_match_count": 0,
                "ransac_inlier_count": 0,
                "inlier_ratio": 0.0,
                "superpoint_ms": 0.0,
                "lightglue_ms": 0.0,
                "ransac_ms": 0.0,
                "fallback_total_ms": 0.0,
                "failure_reason": None,
            }
            return updated_dict, debug_info
        
        logger.debug(f"Frame {self.current_frame_index}: missing markers {missing_ids}")
        
        # SAFETY GUARDRAIL: Limit number of fallback attempts per frame
        max_attempts = self.cfg.max_fallback_markers_per_frame
        missing_ids_list = list(missing_ids)
        
        # Prioritize markers by recency (most recently seen first)
        missing_ids_sorted = sorted(
            missing_ids_list,
            key=lambda mid: self.tracker_states.get(mid, TrackerState(mid, None, -999)).last_seen_frame_index,
            reverse=True
        )
        
        recovery_attempts = 0
        fallback_success = False
        raw_match_count = 0
        ransac_inlier_count = 0
        superpoint_ms = 0.0
        lightglue_ms = 0.0
        ransac_ms = 0.0
        fallback_total_ms = 0.0
        failure_reason: Optional[str] = None
        
        # Attempt recovery for each missing marker (up to max_attempts)
        for marker_id in missing_ids_sorted:
            if recovery_attempts >= max_attempts:
                logger.debug(f"Reached max fallback attempts ({max_attempts}) for this frame")
                break
            
            recovery_attempts += 1
            attempt_start = time.perf_counter()
            recovered_corners, info, fail_reason = self._recover_marker(
                marker_id, frame_gray, frame_bgr
            )
            fallback_total_ms += (time.perf_counter() - attempt_start) * 1000.0

            if fail_reason == "insufficient_inliers":
                failure_reason = "insufficient_inliers"
            elif failure_reason is None and fail_reason is not None:
                failure_reason = fail_reason

            if info is not None:
                raw_match_count += int(info.raw_match_count)
                ransac_inlier_count += int(info.ransac_inlier_count)
                superpoint_ms += float(info.superpoint_ms)
                lightglue_ms += float(info.lightglue_ms)
                ransac_ms += float(info.ransac_ms)

            if recovered_corners is not None:
                # Verify corners are in correct OpenCV ArUco format: (4, 2) float32
                if not isinstance(recovered_corners, np.ndarray):
                    recovered_corners = np.array(recovered_corners, dtype=np.float32)
                if recovered_corners.shape != (4, 2):
                    logger.warning(f"Invalid corner shape for marker {marker_id}: {recovered_corners.shape}")
                    continue
                if recovered_corners.dtype != np.float32:
                    recovered_corners = recovered_corners.astype(np.float32)
                
                updated_dict[marker_id] = recovered_corners
                debug_info.append(info)
                fallback_success = True
                logger.info(f"Recovered marker {marker_id} via {info.source}")

        inlier_ratio = 0.0
        if raw_match_count > 0:
            inlier_ratio = ransac_inlier_count / float(raw_match_count)

        self._last_frame_stats = {
            "fallback_triggered": True,
            "fallback_success": fallback_success,
            "raw_match_count": raw_match_count,
            "ransac_inlier_count": ransac_inlier_count,
            "inlier_ratio": inlier_ratio,
            "superpoint_ms": superpoint_ms,
            "lightglue_ms": lightglue_ms,
            "ransac_ms": ransac_ms,
            "fallback_total_ms": fallback_total_ms,
            "failure_reason": failure_reason,
        }
        
        return updated_dict, debug_info

    def get_last_frame_stats(self) -> dict[str, Any]:
        """Return fallback evaluation stats collected for the most recent frame."""
        return dict(self._last_frame_stats)
    
    def _recover_marker(
        self,
        marker_id: int,
        frame_gray: np.ndarray,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo], Optional[str]]:
        """Attempt to recover a single missing marker.
        
        Strategy:
        1. If marker was recently seen, try tracking
        2. If tracking fails or marker not recent, check reacquire interval
        3. Try template reacquire if interval allows
        
        Args:
            marker_id: ID of marker to recover
            frame_gray: Current frame in grayscale
            frame_bgr: Current frame in BGR
        
        Returns:
            (corners, debug_info) or (None, None) if recovery failed
        """
        # Try tracking first if marker was recently seen
        tracker_state = self.tracker_states.get(marker_id)
        if tracker_state is not None:
            age = self.current_frame_index - tracker_state.last_seen_frame_index
            if age <= self.cfg.max_age_frames and tracker_state.last_frame_gray is not None:
                corners, info = self._track_marker(
                    marker_id, tracker_state, frame_gray
                )
                if corners is not None:
                    # Verify ID
                    if self._verify_marker_id(frame_gray, corners, marker_id):
                        return corners, info, None
                    else:
                        logger.debug(f"Tracking result failed ID verification for marker {marker_id}")
        
        # SAFETY GUARDRAIL: Check reacquire interval
        last_reacquire = self.last_reacquire_frame.get(marker_id, -999)
        frames_since_reacquire = self.current_frame_index - last_reacquire
        
        if frames_since_reacquire < self.cfg.reacquire_interval_frames:
            logger.debug(
                f"Skipping reacquire for marker {marker_id}: only {frames_since_reacquire} frames "
                f"since last attempt (min interval: {self.cfg.reacquire_interval_frames})"
            )
            return None, None, "no_match"
        
        # Fall back to template reacquire
        corners, info, fail_reason = self._reacquire_from_template(marker_id, frame_gray, frame_bgr)
        
        if corners is not None:
            # Update last reacquire frame
            self.last_reacquire_frame[marker_id] = self.current_frame_index
            
            # Verify ID
            if self._verify_marker_id(frame_gray, corners, marker_id):
                return corners, info, None
            else:
                logger.warning(f"Template reacquire result failed ID verification for marker {marker_id}")
                return None, info, "no_match"
        
        return None, info, fail_reason
    
    def _track_marker(
        self,
        marker_id: int,
        tracker_state: TrackerState,
        frame_gray: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo]]:
        """Track marker from last known position using optical flow in ROI.
        
        Args:
            marker_id: ID of marker to track
            tracker_state: Last known state
            frame_gray: Current frame
        
        Returns:
            (corners, debug_info) or (None, None) if tracking failed
        """
        flow_start = time.perf_counter()
        try:
            # Define ROI around last known position
            last_corners = tracker_state.last_corners
            x_min = int(np.floor(last_corners[:, 0].min())) - self.cfg.roi_expand_px
            x_max = int(np.ceil(last_corners[:, 0].max())) + self.cfg.roi_expand_px
            y_min = int(np.floor(last_corners[:, 1].min())) - self.cfg.roi_expand_px
            y_max = int(np.ceil(last_corners[:, 1].max())) + self.cfg.roi_expand_px
            
            h, w = frame_gray.shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                return None, None
            
            # Extract ROIs
            prev_roi = tracker_state.last_frame_gray[y_min:y_max, x_min:x_max]
            curr_roi = frame_gray[y_min:y_max, x_min:x_max]
            
            # Adjust corners to ROI coordinates
            roi_corners = last_corners.copy()
            roi_corners[:, 0] -= x_min
            roi_corners[:, 1] -= y_min
            
            # Use optical flow to track corners
            next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                prev_roi,
                curr_roi,
                roi_corners.astype(np.float32),
                None,
                winSize=(21, 21),
                maxLevel=3
            )
            
            # Check if all corners were successfully tracked
            if next_corners is None or status is None or not np.all(status):
                return None, None
            
            # Convert back to image coordinates
            tracked_corners = next_corners.copy()
            tracked_corners[:, 0] += x_min
            tracked_corners[:, 1] += y_min
            
            # Sanity check: ensure corners are within frame
            if (tracked_corners < 0).any() or \
               (tracked_corners[:, 0] >= w).any() or \
               (tracked_corners[:, 1] >= h).any():
                return None, None
            
            # Refine corners if enabled
            if self.cfg.corner_refine:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                tracked_corners = cv2.cornerSubPix(
                    frame_gray,
                    tracked_corners,
                    (5, 5),
                    (-1, -1),
                    criteria
                )
            
            info = DebugInfo(
                marker_id=marker_id,
                source="lg_track",
                inliers=4,
                match_quality=1.0 - err.mean() if err is not None else 1.0,
                raw_match_count=4,
                ransac_inlier_count=4,
                inlier_ratio=1.0,
                fallback_total_ms=(time.perf_counter() - flow_start) * 1000.0,
            )
            
            return tracked_corners.reshape(4, 2).astype(np.float32), info
            
        except Exception as e:
            logger.debug(f"Tracking failed for marker {marker_id}: {e}")
            return None, None
    
    def _reacquire_from_template(
        self,
        marker_id: int,
        frame_gray: np.ndarray,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo], Optional[str]]:
        """Re-acquire marker by matching template using LightGlue.
        
        Uses cached template features (computed once at init) to avoid recomputation.
        Prefers ROI-based matching if last_corners exist, otherwise full frame.
        
        Args:
            marker_id: ID of marker to recover
            frame_gray: Current frame in grayscale
            frame_bgr: Current frame in BGR
        
        Returns:
            (corners, debug_info) or (None, None) if reacquire failed
        """
        logger.info(f"Attempting LightGlue match for marker_id={marker_id}, frame={self.current_frame_index}")
        
        if marker_id not in self.templates:
            logger.warning(f"Skipping match: no template found for marker {marker_id}")
            return None, None, "no_match"
        
        total_start = time.perf_counter()
        try:
            import torch
            
            template_data = self.templates[marker_id]
            template_kpts = template_data['keypoints']  # Cached torch tensor (N, 2)
            template_desc = template_data['descriptors']  # Cached torch tensor (N, 256)
            template_corners = template_data['corners']
            
            # ALWAYS use tight ROI around marker to reduce noise
            # Priority 1: Use last-known marker position if available
            # Priority 2: Crop central region as fallback
            search_region = frame_gray
            roi_offset_x = 0
            roi_offset_y = 0
            h_full, w_full = frame_gray.shape
            
            # Try to get ROI from last known marker position
            roi_found = False
            if marker_id in self.tracker_states:
                tracker_state = self.tracker_states[marker_id]
                if tracker_state.last_corners is not None:
                    # ROI around last known position with tight expansion
                    roi_expand = self.cfg.roi_expand_px
                    last_corners = tracker_state.last_corners
                    
                    x_min = int(np.floor(last_corners[:, 0].min())) - roi_expand
                    x_max = int(np.ceil(last_corners[:, 0].max())) + roi_expand
                    y_min = int(np.floor(last_corners[:, 1].min())) - roi_expand
                    y_max = int(np.ceil(last_corners[:, 1].max())) + roi_expand
                    
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w_full, x_max)
                    y_max = min(h_full, y_max)
                    
                    if x_max > x_min and y_max > y_min:
                        search_region = frame_gray[y_min:y_max, x_min:x_max]
                        roi_offset_x = x_min
                        roi_offset_y = y_min
                        roi_found = True
                        logger.info(f"[LightGlue] Using tight ROI from last-known position for marker {marker_id}: ({x_min},{y_min})-({x_max},{y_max}), region size {x_max-x_min}x{y_max-y_min}")
            
            # Fallback: Crop central region if no tracker state
            if not roi_found:
                # Default to central region around frame center (common marker location)
                # Crop a 60% x 60% region centered, maintaining aspect ratio
                crop_w = max(int(w_full * 0.6), 300)
                crop_h = max(int(h_full * 0.6), 300)
                
                x_min = (w_full - crop_w) // 2
                y_min = (h_full - crop_h) // 2
                x_max = x_min + crop_w
                y_max = y_min + crop_h
                
                search_region = frame_gray[y_min:y_max, x_min:x_max]
                roi_offset_x = x_min
                roi_offset_y = y_min
                logger.info(f"[LightGlue] Using central ROI region for marker {marker_id}: ({x_min},{y_min})-({x_max},{y_max}), region size {x_max-x_min}x{y_max-y_min}")
            
            # Extract features from search region
            h, w = search_region.shape
            frame_tensor = torch.from_numpy(search_region.astype(np.float32) / 255.0).to(self.device)[None, None]
            superpoint_start = time.perf_counter()
            
            with torch.no_grad():
                frame_feats = self.superpoint({'image': frame_tensor})
                frame_kpts = frame_feats['keypoints'][0]  # (N, 2) on device
                frame_desc = frame_feats['descriptors'][0].T  # (N, 256) on device
            superpoint_ms = (time.perf_counter() - superpoint_start) * 1000.0
            logger.info(f"SuperPoint extracted {len(frame_kpts)} keypoints for marker {marker_id}")

            # Call LightGlue with extracted SuperPoint features
            # Prepare features dict for LightGlue with correct shapes:
            # LightGlue expects:
            #   keypoints shape (batch, num_kpts, 2)
            #   descriptors shape (batch, 256, num_kpts) - SuperPoint native format
            feats0 = {
                'keypoints': template_kpts[None],  # Add batch dim: (1, N, 2)
                'descriptors': template_desc.T[None]  # Transpose back from (N,256) to (256,N), then batch: (1, 256, N)
            }
            feats1 = {
                'keypoints': frame_kpts[None],  # Add batch dim: (1, M, 2)
                'descriptors': frame_desc.T[None]  # Transpose back from (N,256) to (256,M), then batch: (1, 256, M)
            }

            # Log before calling LightGlue
            logger.info(f"[LightGlue] Running match with {template_kpts.shape[0]} template keypoints and {frame_kpts.shape[0]} frame keypoints for marker {marker_id}")

            # Match features using LightGlue with correct format (image0/image1 keys)
            lightglue_start = time.perf_counter()
            matches = self.lightglue({
                "image0": feats0,
                "image1": feats1
            })
            lightglue_ms = (time.perf_counter() - lightglue_start) * 1000.0
            
            # Extract matched keypoints using matches0
            matches0 = matches['matches0']  # Shape (num_keypoints_template,)
            valid = matches0 >= 0  # Boolean mask for valid matches
            raw_match_count = int(valid.sum())
            logger.info(f"LightGlue produced raw_match_count={raw_match_count} for marker {marker_id}")
            
            template_matched_idx = torch.where(valid)[0]  # Indices of matched template keypoints
            frame_matched_idx = matches0[valid]  # Indices of matched frame keypoints
            
            if len(template_matched_idx) > 0:
                match_pairs = torch.stack([template_matched_idx, frame_matched_idx], dim=1)
            else:
                match_pairs = torch.empty((0, 2), device=template_matched_idx.device, dtype=torch.long)

            # Save correspondence image even if matching failed
            self._save_correspondence_debug_image(
                marker_id=marker_id,
                template_image=template_data["image"],
                frame_image=search_region,
                template_kpts=template_kpts,
                frame_kpts=frame_kpts,
                match_pairs=match_pairs,
                frame_index=self.current_frame_index,
                label="reacquire",
                raw_match_count=raw_match_count,
            )
            
            if valid.sum() < self.cfg.min_inliers:
                logger.warning(f"Skipping match: not enough keypoints for marker {marker_id}: got {valid.sum()}, need {self.cfg.min_inliers}")
                info = DebugInfo(
                    marker_id=marker_id,
                    source="lg_reacquire",
                    inliers=0,
                    match_quality=0.0,
                    raw_match_count=raw_match_count,
                    ransac_inlier_count=0,
                    inlier_ratio=0.0,
                    superpoint_ms=superpoint_ms,
                    lightglue_ms=lightglue_ms,
                    ransac_ms=0.0,
                    fallback_total_ms=(time.perf_counter() - total_start) * 1000.0,
                    success=False,
                    failure_reason="insufficient_inliers",
                )
                return None, info, "insufficient_inliers"
            
            # Extract matched point coordinates (convert to numpy for homography)
            template_pts = template_kpts[template_matched_idx].cpu().numpy()  # (M, 2)
            frame_pts = frame_kpts[frame_matched_idx].cpu().numpy()  # (M, 2)
            
            # Translate frame_pts back to full-frame coordinates (they're currently in ROI space)
            frame_pts = frame_pts + np.array([[roi_offset_x, roi_offset_y]], dtype=np.float32)
            logger.info(f"[LightGlue] Translated {len(frame_pts)} matched points from ROI space to full-frame coordinates (offset: {roi_offset_x}, {roi_offset_y})")
            
            # Compute homography using ONLY matched keypoints
            ransac_start = time.perf_counter()
            H, mask = cv2.findHomography(
                template_pts,
                frame_pts,
                cv2.RANSAC,
                5.0
            )
            ransac_ms = (time.perf_counter() - ransac_start) * 1000.0
            
            if H is None:
                logger.warning(f"Skipping match: homography computation failed for marker {marker_id}")
                info = DebugInfo(
                    marker_id=marker_id,
                    source="lg_reacquire",
                    inliers=0,
                    match_quality=0.0,
                    raw_match_count=raw_match_count,
                    ransac_inlier_count=0,
                    inlier_ratio=0.0,
                    superpoint_ms=superpoint_ms,
                    lightglue_ms=lightglue_ms,
                    ransac_ms=ransac_ms,
                    fallback_total_ms=(time.perf_counter() - total_start) * 1000.0,
                    success=False,
                    failure_reason="no_match",
                )
                return None, info, "no_match"
            
            inliers = mask.sum() if mask is not None else 0
            if inliers < self.cfg.min_inliers:
                logger.warning(f"Skipping match: insufficient RANSAC inliers for marker {marker_id}: got {inliers}, need {self.cfg.min_inliers}")
                info = DebugInfo(
                    marker_id=marker_id,
                    source="lg_reacquire",
                    inliers=int(inliers),
                    match_quality=0.0,
                    raw_match_count=raw_match_count,
                    ransac_inlier_count=int(inliers),
                    inlier_ratio=(int(inliers) / float(raw_match_count)) if raw_match_count > 0 else 0.0,
                    superpoint_ms=superpoint_ms,
                    lightglue_ms=lightglue_ms,
                    ransac_ms=ransac_ms,
                    fallback_total_ms=(time.perf_counter() - total_start) * 1000.0,
                    success=False,
                    failure_reason="insufficient_inliers",
                )
                return None, info, "insufficient_inliers"
            
            # Project template corners to frame (within search region)
            corners_transformed = cv2.perspectiveTransform(
                template_corners.reshape(1, 4, 2),
                H
            )
            recovered_corners = corners_transformed.reshape(4, 2).astype(np.float32)
            
            # Adjust corners back to full frame coordinates if ROI was used
            if roi_offset_x != 0 or roi_offset_y != 0:
                recovered_corners[:, 0] += roi_offset_x
                recovered_corners[:, 1] += roi_offset_y
            
            # Sanity check: ensure corners are within full frame
            h_full, w_full = frame_gray.shape
            if (recovered_corners < 0).any() or \
               (recovered_corners[:, 0] >= w_full).any() or \
               (recovered_corners[:, 1] >= h_full).any():
                logger.warning(f"Skipping match: recovered corners out of bounds for marker {marker_id}")
                return None, None
            
            # Refine corners if enabled
            if self.cfg.corner_refine:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                recovered_corners = cv2.cornerSubPix(
                    frame_gray,
                    recovered_corners,
                    (5, 5),
                    (-1, -1),
                    criteria
                )
            
            match_quality = inliers / valid.sum() if valid.sum() > 0 else 0.0
            logger.info(f"LightGlue reacquire SUCCESS for marker {marker_id}: {int(inliers)} RANSAC inliers from {raw_match_count} raw matches")
            
            info = DebugInfo(
                marker_id=marker_id,
                source="lg_reacquire",
                inliers=int(inliers),
                match_quality=match_quality,
                homography=H,
                raw_match_count=raw_match_count,
                ransac_inlier_count=int(inliers),
                inlier_ratio=match_quality,
                superpoint_ms=superpoint_ms,
                lightglue_ms=lightglue_ms,
                ransac_ms=ransac_ms,
                fallback_total_ms=(time.perf_counter() - total_start) * 1000.0,
                success=True,
                failure_reason=None,
            )
            
            return recovered_corners, info, None
            
        except Exception as e:
            logger.error(f"Template reacquire exception for marker {marker_id}: {e}", exc_info=True)
            if self.debug_matches_dir is not None:
                try:
                    import torch
                    self._save_correspondence_debug_image(
                        marker_id=marker_id,
                        template_image=template_data["image"] if marker_id in self.templates else np.zeros((100, 100), dtype=np.uint8),
                        frame_image=frame_gray,
                        template_kpts=torch.empty((0, 2)),
                        frame_kpts=torch.empty((0, 2)),
                        match_pairs=torch.empty((0, 2)),
                        frame_index=self.current_frame_index,
                        label="exception",
                        raw_match_count=0,
                    )
                except Exception as save_e:
                    logger.debug(f"Failed to save exception debug image: {save_e}")
            return None, None, "no_match"
