"""Load training data from Postgres or CSV/JSONL file."""
import os
import csv
import json
import logging
from typing import List, Dict, Optional
import psycopg

logger = logging.getLogger(__name__)

LABEL2ID = {"Milik Negara": 0, "Bukan Milik Negara": 1}


def load_from_file(path: str) -> List[Dict]:
    """Load training data from CSV or JSONL file."""
    samples = []
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                label = row.get("label", "").strip()
                if text and label in LABEL2ID:
                    samples.append({"text": text, "label": label})
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get("text", "").strip()
                label = item.get("label", "").strip()
                if text and label in LABEL2ID:
                    samples.append({"text": text, "label": label})
    else:
        raise ValueError(f"Unsupported file format: {path}")

    logger.info("Loaded %d samples from file: %s", len(samples), path)
    return samples


def load_from_postgres(database_url: str) -> List[Dict]:
    """Load approved records from Postgres."""
    query = """
        SELECT text, final_label
        FROM records_gratifikasirecord
        WHERE final_label IS NOT NULL
          AND final_label IN ('Milik Negara', 'Bukan Milik Negara')
    """
    samples = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            for text, label in rows:
                if text and label:
                    samples.append({"text": text.strip(), "label": label.strip()})

    logger.info("Loaded %d samples from Postgres", len(samples))
    return samples


def load_training_data(
    database_url: Optional[str] = None,
    train_data_path: Optional[str] = None,
) -> List[Dict]:
    """Load training data with precedence: file > Postgres."""
    if train_data_path and os.path.exists(train_data_path):
        logger.info("Using training data from file: %s", train_data_path)
        return load_from_file(train_data_path)

    if database_url:
        logger.info("Loading training data from Postgres")
        return load_from_postgres(database_url)

    raise ValueError(
        "No training data source available. "
        "Set TRAIN_DATA_PATH or DATABASE_URL."
    )
