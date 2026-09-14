from camera_host.manager import CameraHardwareService, build_hardware_factory


def test_host_service_session_lifecycle_with_fake_backend():
    hardware_factory = build_hardware_factory("fake", device=None, fps=25, width=64, height=48)
    service = CameraHardwareService(
        backend_name="fake",
        hardware_factory=hardware_factory,
        jpeg_quality=80,
    )
    service.start()

    stream_id, stream_queue = service.register_stream()
    start = service.start_session("test-session", fps=25, width=64, height=48)
    assert start["session_active"] is True

    sample = stream_queue.get(timeout=2)
    assert sample["frame_idx"] == 1
    assert sample["session_id"] == "test-session"
    assert sample["capture_time_ms"] > 0
    assert "image_b64" in sample

    latest = service.latest_frame()
    assert latest is not None
    assert latest["frame_idx"] == sample["frame_idx"]

    stop = service.stop_session()
    assert stop["session_active"] is False

    service.unregister_stream(stream_id)
    service.close()
