from iiot_pipeline.config import RunConfig, SessionSummary


def test_runconfig_initialization():
    """Ensure explicit values are preserved when constructing RunConfig."""
    cfg = RunConfig(
        fps=30,
        grayscale=True,
        calibrationPath="calib/test.yml",
        sessionRoot="sessions",
        durationSec=5.0,
        arucoDict="4x4_50",
        markerLengthM=0.04,
        device=0,
        targetIds=[1, 2, 3],
    )

    assert cfg.fps == 30
    assert cfg.grayscale is True
    assert cfg.targetIds == [1, 2, 3]
    assert cfg.sessionRoot == "sessions"


def test_runconfig_defaults_cover_all_keys():
    """Verify defaults match expected sane values."""
    cfg = RunConfig()
    assert cfg.fps == 15
    assert cfg.grayscale is False
    assert cfg.calibrationPath.endswith(".yml")
    assert cfg.sessionRoot == "data/sessions"
    assert cfg.targetIds is None


def test_session_summary_is_simple_container():
    """Validate SessionSummary stores and exposes simple fields."""
    summary = SessionSummary("session", 10, "csv", "log", 8.2, 0)
    assert summary.sessionPath == "session"
    assert summary.framesProcessed == 10
    assert summary.csvPath == "csv"
