import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np


TABLES = ("tasks", "experiments", "agent_traces", "tool_calls", "artifacts",
          "iq_metrics", "optimization_trials", "reports")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / "files"
        self.files.mkdir(exist_ok=True)
        self.database = self.root / "lab.sqlite3"
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            for table in TABLES:
                connection.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                      id TEXT PRIMARY KEY, experiment_id TEXT,
                      created REAL NOT NULL, payload TEXT NOT NULL
                    )
                """)
                connection.execute(f"CREATE INDEX IF NOT EXISTS ix_{table} ON {table}(experiment_id)")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, table, payload, experiment=None, key=None):
        self._table(table)
        key = key or uuid4().hex
        with self.connection() as connection:
            connection.execute(f"""
                INSERT INTO {table} VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
            """, (key, experiment, time.time(), json_text(payload)))
        return key

    def rows(self, table, experiment=None):
        self._table(table)
        query = f"SELECT * FROM {table}"
        args = ()
        if experiment is not None:
            query += " WHERE experiment_id=?"
            args = (experiment,)
        with self.connection() as connection:
            return [self._decode(row) for row in connection.execute(query + " ORDER BY created", args)]

    def get(self, table, key):
        self._table(table)
        with self.connection() as connection:
            row = connection.execute(f"SELECT * FROM {table} WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown {table} ID: {key}")
        return self._decode(row)

    @staticmethod
    def _decode(row):
        return {**dict(row), "payload": json.loads(row["payload"])}

    @staticmethod
    def _table(table):
        if table not in TABLES:
            raise ValueError("Unknown table")

    def put(self, content, kind, metadata=None, experiment=None, suffix=".bin"):
        if suffix not in {".bin", ".npy", ".dng", ".png", ".html", ".csv", ".json"}:
            raise ValueError("Unsupported artifact suffix")
        key = uuid4().hex
        path = self.files / (key + suffix)
        try:
            with path.open("xb") as stream:
                if suffix == ".npy":
                    np.save(stream, content, allow_pickle=False)
                else:
                    stream.write(content)
            self.record("artifacts", {"file": path.name, "kind": kind,
                                      "metadata": metadata or {}}, experiment, key)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return key

    def path(self, key):
        name = self.get("artifacts", key)["payload"]["file"]
        path = (self.files / name).resolve()
        if path.parent != self.files.resolve() or not path.is_file():
            raise ValueError("Invalid artifact path")
        return path

    def array(self, key):
        path = self.path(key)
        if path.suffix != ".npy":
            raise ValueError("Expected a NumPy artifact")
        return np.load(path, allow_pickle=False)

