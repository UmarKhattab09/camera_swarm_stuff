"""Basic unit tests for LightGlue fallback configuration and structure."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_fusion.config import LightGlueConfig, CameraConfig, load_config
import json
import tempfile


def test_lightglue_config_defaults():
    """Test LightGlueConfig default values."""
    cfg = LightGlueConfig()
    assert cfg.enabled is False
    assert cfg.device == "cpu"
    assert cfg.template_dir == "templates/markers"
    assert cfg.min_inliers == 4
    assert cfg.max_age_frames == 5
    assert cfg.roi_expand_px == 50
    assert cfg.debug_save is False
    assert cfg.corner_refine is True
    assert cfg.match_threshold == 0.2
    assert cfg.debug_matches_dir is None
    print("✓ LightGlueConfig defaults correct")


def test_lightglue_config_dict():
    """Test LightGlueConfig serialization."""
    cfg = LightGlueConfig(enabled=True, device="cuda", min_inliers=6)
    d = cfg.as_dict()
    assert d["enabled"] is True
    assert d["device"] == "cuda"
    assert d["min_inliers"] == 6
    print("✓ LightGlueConfig serialization works")


def test_camera_config_with_lightglue():
    """Test CameraConfig with LightGlue section."""
    cfg = CameraConfig()
    cfg.lightglue = LightGlueConfig(enabled=True)
    assert cfg.lightglue is not None
    assert cfg.lightglue.enabled is True
    print("✓ CameraConfig can include LightGlueConfig")


def test_load_config_with_lightglue():
    """Test loading config from JSON with LightGlue section."""
    config_data = {
        "camera_name": "test_cam",
        "device": 0,
        "lightglue": {
            "enabled": True,
            "device": "cuda",
            "template_dir": "custom/templates",
            "min_inliers": 6,
            "max_age_frames": 10
        }
    }
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        cfg = load_config(temp_path)
        assert cfg.camera_name == "test_cam"
        assert cfg.lightglue is not None
        assert cfg.lightglue.enabled is True
        assert cfg.lightglue.device == "cuda"
        assert cfg.lightglue.template_dir == "custom/templates"
        assert cfg.lightglue.min_inliers == 6
        assert cfg.lightglue.max_age_frames == 10
        # Check defaults are preserved for non-specified fields
        assert cfg.lightglue.roi_expand_px == 50
        assert cfg.lightglue.debug_save is False
        assert cfg.lightglue.debug_matches_dir is None
        print("✓ Config loading with LightGlue section works")
    finally:
        Path(temp_path).unlink()


def test_load_config_without_lightglue():
    """Test loading config without LightGlue section."""
    config_data = {
        "camera_name": "test_cam",
        "device": 0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        cfg = load_config(temp_path)
        assert cfg.camera_name == "test_cam"
        assert cfg.lightglue is None
        print("✓ Config loading without LightGlue section works")
    finally:
        Path(temp_path).unlink()


def test_camera_configs_use_separate_template_dirs():
    """Test the checked-in camera configs point at camera-specific templates."""
    cam1_cfg = load_config("configs/cam1.json")
    cam2_cfg = load_config("configs/cam2.json")

    assert cam1_cfg.lightglue is not None
    assert cam2_cfg.lightglue is not None
    assert cam1_cfg.lightglue.enabled is True
    assert cam2_cfg.lightglue.enabled is True
    assert cam1_cfg.lightglue.template_dir == "templates/markers/cam1"
    assert cam2_cfg.lightglue.template_dir == "templates/markers/cam2"


def test_lightglue_fallback_import():
    """Test that LightGlueFallback can be imported."""
    try:
        from camera_fusion.fallback import LightGlueFallback
        print("✓ LightGlueFallback can be imported")
    except ImportError as e:
        print(f"✗ Failed to import LightGlueFallback: {e}")
        raise


def test_lightglue_fallback_graceful_init():
    """Test LightGlueFallback gracefully handles missing dependencies."""
    from camera_fusion.fallback import LightGlueFallback
    
    cfg = LightGlueConfig(enabled=True, device="cpu")
    
    # This should not crash even if torch/lightglue are missing
    try:
        fallback = LightGlueFallback(cfg, "4x4_50")
        # If dependencies are missing, it should disable itself
        if not fallback.torch_available:
            assert fallback.enabled is False
            print("✓ LightGlueFallback gracefully handles missing PyTorch")
        else:
            print("✓ LightGlueFallback initialized with PyTorch available")
    except Exception as e:
        # Should not raise exceptions, but log warnings
        print(f"✗ LightGlueFallback raised unexpected exception: {e}")
        raise


def test_lightglue_fallback_disabled():
    """Test LightGlueFallback with disabled config."""
    from camera_fusion.fallback import LightGlueFallback
    
    cfg = LightGlueConfig(enabled=False)
    fallback = LightGlueFallback(cfg, "4x4_50")
    
    assert fallback.enabled is False
    assert fallback.templates == {}
    print("✓ LightGlueFallback respects disabled config")


def main():
    """Run all tests."""
    print("="*60)
    print("Running LightGlue Fallback Configuration Tests")
    print("="*60)
    print()
    
    tests = [
        test_lightglue_config_defaults,
        test_lightglue_config_dict,
        test_camera_config_with_lightglue,
        test_load_config_with_lightglue,
        test_load_config_without_lightglue,
        test_camera_configs_use_separate_template_dirs,
        test_lightglue_fallback_import,
        test_lightglue_fallback_graceful_init,
        test_lightglue_fallback_disabled,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
