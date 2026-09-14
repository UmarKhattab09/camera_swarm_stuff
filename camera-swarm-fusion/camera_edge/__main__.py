import json
import sys
import time

from .agent import EdgeAgent
from .config import EdgeAgentConfig
from .health import EdgeHealthTracker


def _healthcheck(config: EdgeAgentConfig) -> int:
    tracker = EdgeHealthTracker(config.health_path)

    try:
        payload = tracker.read()
    except FileNotFoundError:
        return 1

    updated_at_ms = int(payload.get("updated_at_ms", 0))
    age_ms = int(time.time_ns() / 1e6) - updated_at_ms
    if age_ms > config.stale_after_s * 1000:
        return 1

    return 0 if payload.get("state") in {"streaming", "waiting"} else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    config = EdgeAgentConfig.from_env()

    if argv and argv[0] == "healthcheck":
        return _healthcheck(config)

    if argv and argv[0] == "print-config":
        print(json.dumps(config.__dict__, indent=2, sort_keys=True, default=str))
        return 0

    agent = EdgeAgent(config)
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
