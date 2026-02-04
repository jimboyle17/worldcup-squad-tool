import json
import sqlite3
import time
from pathlib import Path
from typing import Optional


class ScraperCache:
    """SQLite-based caching layer for scraped data."""

    def __init__(self, db_path: str, expiry_hours: int = 24):
        self.db_path = db_path
        self.expiry_seconds = expiry_hours * 3600
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Retrieve cached data if it exists and hasn't expired."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data, timestamp FROM cache WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            return None

        data, timestamp = row
        if time.time() - timestamp > self.expiry_seconds:
            self.delete(key)
            return None

        return data

    def set(self, key: str, data: str):
        """Store data in cache with current timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, timestamp) VALUES (?, ?, ?)",
                (key, data, time.time()),
            )
            conn.commit()

    def delete(self, key: str):
        """Remove a specific cache entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        """Clear all cached data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def get_json(self, key: str) -> Optional[list]:
        data = self.get(key)
        if data is not None:
            return json.loads(data)
        return None

    def set_json(self, key: str, data: list):
        self.set(key, json.dumps(data))
