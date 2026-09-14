from unittest.mock import MagicMock, patch

from iiot_pipeline.cli import main


def test_main_builds_facade_and_runs():
    """Ensure CLI wiring builds all components and runs the facade once."""
    fake_facade = MagicMock()
    fake_facade.run_session.return_value = "summary"

    with patch(
        "iiot_pipeline.cli.StrategyFactory.from_config",
        return_value=("cap", "pre", "und", "det", "loc"),
    ) as mock_factory, patch(
        "iiot_pipeline.cli.SessionStorage"
    ) as mock_storage, patch(
        "iiot_pipeline.cli.CameraPipelineFacade", return_value=fake_facade
    ) as mock_facade:
        mock_storage.return_value = MagicMock()
        argv = [
            "prog",
            "--fps",
            "10",
            "--duration",
            "0.1",
            "--target-ids",
            "1",
            "2",
        ]
        with patch("sys.argv", argv):
            main()

    mock_factory.assert_called_once()
    mock_storage.assert_called_once()
    mock_facade.assert_called_once()
    fake_facade.run_session.assert_called_once()
