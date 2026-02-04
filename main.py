"""World Cup 2026 Squad Assessment Tool - Entry Point."""

import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = Path(__file__).resolve().parent

    # Load config
    config_path = base_dir / "config.json"
    if config_path.exists():
        config = load_json(str(config_path))
    else:
        config = {
            "app_name": "World Cup 2026 Squad Assessment Tool",
            "cache_expiry_hours": 24,
            "scrape_delay_seconds": 2,
            "window_width": 1280,
            "window_height": 800,
        }

    # Load team metadata
    teams_path = base_dir / "data" / "teams_2026.json"
    if not teams_path.exists():
        logger.error(f"Teams metadata not found at {teams_path}")
        sys.exit(1)

    teams_data = load_json(str(teams_path))
    teams_meta = teams_data.get("teams", [])
    logger.info(f"Loaded metadata for {len(teams_meta)} teams")

    # Load manager metadata
    managers_path = base_dir / "data" / "managers_2026.json"
    managers_meta = []
    if managers_path.exists():
        managers_data = load_json(str(managers_path))
        managers_meta = managers_data.get("managers", [])
        logger.info(f"Loaded metadata for {len(managers_meta)} managers")
    else:
        logger.warning(f"Managers metadata not found at {managers_path}")

    # Launch GUI
    from src.gui.app import App

    app = App(config=config, teams_meta=teams_meta, managers_meta=managers_meta)
    app.mainloop()


if __name__ == "__main__":
    main()
