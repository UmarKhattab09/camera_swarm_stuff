from .config import HostServiceConfig
from .manager import CameraHardwareService, ServiceError, build_hardware_factory
from .server import run_server


def main() -> None:
    config = HostServiceConfig.from_env()
    hardware_factory = build_hardware_factory(
        config.backend, config.device, config.fps, config.width, config.height
    )
    service = CameraHardwareService(
        backend_name=config.backend,
        hardware_factory=hardware_factory,
        jpeg_quality=config.jpeg_quality,
        reconnect_delay_s=config.reconnect_delay_s,
        stream_queue_size=config.stream_queue_size,
    )
    service.start()

    if config.startup_session:
        try:
            service.start_session(config.startup_session_id, config.fps, config.width, config.height)
        except ServiceError:
            pass

    try:
        run_server(config.socket_path, service)
    finally:
        service.close()


if __name__ == "__main__":
    main()
