"""Idea Lake — SQLite archive for completed/failed ideas.

Provides a queryable store so ideas.md can stay small (~500 hot ideas)
while all historical ideas remain accessible for config lookups, dedup,
and leaderboard queries.

Usage:
    lake = IdeaLake("idea_lake.db")
    lake.insert("idea-001", "Zipformer", config_yaml, raw_md, metrics={...})
    idea = lake.get("idea-001")
    top = lake.get_top_models(metric="fedex_auc", n=10)
"""

import datetime
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger("idea_lake")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ideas (
    idea_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    category TEXT DEFAULT 'architecture',
    parent TEXT,
    hypothesis TEXT,
    config TEXT NOT NULL,
    raw_markdown TEXT NOT NULL,
    backbone_name TEXT,
    freeze_backbone INTEGER,
    temporal_type TEXT,
    learning_rate REAL,
    num_frames INTEGER,
    fedex_auc REAL,
    fedex_f1 REAL,
    fedex_fpr REAL,
    fedex_fp INTEGER,
    fedex_fn INTEGER,
    nexar_auc REAL,
    status TEXT DEFAULT 'archived',
    training_time REAL,
    archived_at TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_backbone ON ideas(backbone_name);
CREATE INDEX IF NOT EXISTS idx_temporal ON ideas(temporal_type);
CREATE INDEX IF NOT EXISTS idx_fedex_auc ON ideas(fedex_auc DESC);
CREATE INDEX IF NOT EXISTS idx_nexar_auc ON ideas(nexar_auc DESC);
CREATE INDEX IF NOT EXISTS idx_status ON ideas(status);

CREATE TABLE IF NOT EXISTS id_sequence (
    next_id INTEGER NOT NULL
);
"""


def _extract_denormalized(config: dict) -> dict:
    """Extract commonly-queried fields from a YAML config dict.

    Handles both config formats:
      - Old: backbone.name, backbone.freeze, temporal_encoder.type, optimizer.learning_rate
      - New: model.backbone_name, model.freeze_backbone, model.temporal_type, training.lr
    """
    if not isinstance(config, dict):
        return {}
    model = config.get("model", {}) or {}
    backbone = config.get("backbone", {}) or {}
    temporal_enc = config.get("temporal_encoder", {}) or {}
    training = config.get("training", {}) or {}
    optimizer = config.get("optimizer", {}) or {}
    data = config.get("data", {}) or {}
    return {
        "backbone_name": (
            backbone.get("name")
            or model.get("backbone_name")
            or model.get("backbone")
        ),
        "freeze_backbone": 1 if (
            backbone.get("freeze")
            or model.get("freeze_backbone")
        ) else 0,
        "temporal_type": (
            temporal_enc.get("type")
            or model.get("temporal_type")
            or model.get("temporal_model")
        ),
        "learning_rate": (
            optimizer.get("learning_rate")
            or optimizer.get("lr")
            or training.get("learning_rate")
            or training.get("lr")
        ),
        "num_frames": (
            training.get("sequence_length")
            or data.get("frame_sampling", {}).get("max_frames")
        ),
    }


class IdeaLake:
    """SQLite-backed archive for ideas."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        self._ensure_schema()

    def _ensure_schema(self):
        self.conn.executescript(_SCHEMA_SQL)
        # Ensure id_sequence has a row
        row = self.conn.execute("SELECT next_id FROM id_sequence LIMIT 1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO id_sequence (next_id) VALUES (1)")
            self.conn.commit()

    def insert(
        self,
        idea_id: str,
        title: str,
        config_yaml: str,
        raw_markdown: str,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "archived",
    ):
        """Insert or update an idea in the lake."""
        try:
            config = yaml.safe_load(config_yaml) or {}
        except yaml.YAMLError:
            config = {}

        denorm = _extract_denormalized(config)
        metrics = metrics or {}

        self.conn.execute(
            """INSERT OR REPLACE INTO ideas (
                idea_id, title, priority, category, parent, hypothesis,
                config, raw_markdown,
                backbone_name, freeze_backbone, temporal_type,
                learning_rate, num_frames,
                fedex_auc, fedex_f1, fedex_fpr, fedex_fp, fedex_fn,
                nexar_auc,
                status, training_time, archived_at, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?,
                ?, ?, ?, ?
            )""",
            (
                idea_id,
                title,
                metrics.get("priority", "medium"),
                metrics.get("category", "architecture"),
                metrics.get("parent"),
                metrics.get("hypothesis"),
                config_yaml,
                raw_markdown,
                denorm.get("backbone_name"),
                denorm.get("freeze_backbone"),
                denorm.get("temporal_type"),
                denorm.get("learning_rate"),
                denorm.get("num_frames"),
                metrics.get("fedex_auc"),
                metrics.get("fedex_f1"),
                metrics.get("fedex_fpr"),
                metrics.get("fedex_fp"),
                metrics.get("fedex_fn"),
                metrics.get("nexar_auc"),
                status,
                metrics.get("training_time"),
                datetime.datetime.now().isoformat(),
                metrics.get("created_at"),
            ),
        )
        self.conn.commit()

    def get(self, idea_id: str) -> Optional[dict]:
        """Get a single idea by ID."""
        row = self.conn.execute(
            "SELECT * FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def has(self, idea_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()
        return row[0]

    def get_all_ids(self) -> Set[str]:
        """Return set of all idea IDs in the lake."""
        rows = self.conn.execute("SELECT idea_id FROM ideas").fetchall()
        return {r[0] for r in rows}

    def get_max_id_num(self) -> int:
        """Return the highest numeric idea ID in the lake."""
        row = self.conn.execute(
            "SELECT idea_id FROM ideas ORDER BY "
            "CAST(REPLACE(idea_id, 'idea-', '') AS INTEGER) DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0
        return int(row[0].replace("idea-", ""))

    def query(
        self,
        backbone: Optional[str] = None,
        temporal: Optional[str] = None,
        min_auc: Optional[float] = None,
        freeze_only: bool = False,
        limit: int = 20,
    ) -> List[dict]:
        """Filtered query with optional constraints."""
        clauses = []
        params = []
        if backbone:
            clauses.append("backbone_name = ?")
            params.append(backbone)
        if temporal:
            clauses.append("temporal_type = ?")
            params.append(temporal)
        if min_auc is not None:
            clauses.append("fedex_auc >= ?")
            params.append(min_auc)
        if freeze_only:
            clauses.append("freeze_backbone = 1")

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM ideas WHERE {where} "
            f"ORDER BY fedex_auc DESC NULLS LAST LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_models(
        self, metric: str = "fedex_auc", n: int = 20
    ) -> List[dict]:
        """Return top N models by the given metric."""
        if metric not in ("fedex_auc", "nexar_auc", "fedex_f1"):
            metric = "fedex_auc"
        rows = self.conn.execute(
            f"SELECT idea_id, title, backbone_name, temporal_type, "
            f"fedex_auc, fedex_fp, fedex_fn, nexar_auc, status "
            f"FROM ideas WHERE {metric} IS NOT NULL "
            f"ORDER BY {metric} DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_next_id(self) -> int:
        """Atomically get and increment the next idea ID number."""
        cur = self.conn.execute("SELECT next_id FROM id_sequence LIMIT 1")
        row = cur.fetchone()
        next_id = row[0] if row else 1
        self.conn.execute(
            "UPDATE id_sequence SET next_id = ?", (next_id + 1,)
        )
        self.conn.commit()
        return next_id

    def set_next_id(self, n: int):
        """Set the next ID sequence value."""
        self.conn.execute("UPDATE id_sequence SET next_id = ?", (n,))
        self.conn.commit()

    def bulk_insert(self, ideas: List[Dict[str, Any]]):
        """Insert many ideas in a single transaction."""
        for idea in ideas:
            try:
                config_yaml = idea["config_yaml"]
                config = yaml.safe_load(config_yaml) or {}
            except yaml.YAMLError:
                config = {}

            denorm = _extract_denormalized(config)
            metrics = idea.get("metrics", {})

            self.conn.execute(
                """INSERT OR IGNORE INTO ideas (
                    idea_id, title, priority, category, parent, hypothesis,
                    config, raw_markdown,
                    backbone_name, freeze_backbone, temporal_type,
                    learning_rate, num_frames,
                    fedex_auc, fedex_f1, fedex_fpr, fedex_fp, fedex_fn,
                    nexar_auc,
                    status, training_time, archived_at, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?,
                    ?, ?, ?, ?
                )""",
                (
                    idea["idea_id"],
                    idea["title"],
                    idea.get("priority", "medium"),
                    idea.get("category", "architecture"),
                    idea.get("parent"),
                    idea.get("hypothesis"),
                    config_yaml,
                    idea.get("raw_markdown", ""),
                    denorm.get("backbone_name"),
                    denorm.get("freeze_backbone"),
                    denorm.get("temporal_type"),
                    denorm.get("learning_rate"),
                    denorm.get("num_frames"),
                    metrics.get("fedex_auc"),
                    metrics.get("fedex_f1"),
                    metrics.get("fedex_fpr"),
                    metrics.get("fedex_fp"),
                    metrics.get("fedex_fn"),
                    metrics.get("nexar_auc"),
                    idea.get("status", "archived"),
                    metrics.get("training_time"),
                    datetime.datetime.now().isoformat(),
                    metrics.get("created_at"),
                ),
            )
        self.conn.commit()
        logger.info("Bulk inserted %d ideas", len(ideas))

    def close(self):
        self.conn.close()
