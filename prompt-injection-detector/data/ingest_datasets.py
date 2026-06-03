"""
Load deepset/prompt-injections and JasperLS/prompt-injections from HuggingFace,
keep only injection-labelled examples, and upsert them into the ChromaDB
`injection_examples` collection.

Run once from the project root:
    python data/ingest_datasets.py

Pass --reset to delete the collection before re-ingesting.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running as a script from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datasets import load_dataset

from app.core.chroma_client import get_chroma_client
from app.core.config import get_settings

COLLECTION_NAME = "injection_examples"
BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Each entry describes how to extract text + label from a HuggingFace dataset.
DATASET_CONFIGS: list[dict[str, Any]] = [
    {
        "hf_name": "deepset/prompt-injections",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        # Injection is represented as integer 1 in this dataset.
        "injection_values": {1, "1", "INJECTION", "injection"},
    },
    {
        "hf_name": "JasperLS/prompt-injections",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "injection_values": {1, "1", "INJECTION", "injection"},
    },
]


def _make_doc_id(text: str) -> str:
    """Deterministic 16-char hex ID — stable across re-runs, collision-safe for ~50k docs."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _is_injection(value: Any, injection_values: set) -> bool:
    return value in injection_values


def ingest(reset: bool = False) -> None:
    settings = get_settings()
    client = get_chroma_client()

    ef = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
        device="cpu",
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Deleted existing collection '%s'.", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total_upserted = 0

    for cfg in DATASET_CONFIGS:
        hf_name: str = cfg["hf_name"]
        log.info("Loading dataset: %s (split=%s)", hf_name, cfg["split"])

        try:
            dataset = load_dataset(hf_name, split=cfg["split"])
        except Exception as exc:
            log.error("Failed to load %s — skipping. Error: %s", hf_name, exc)
            continue

        injection_rows = [
            row
            for row in dataset
            if _is_injection(row[cfg["label_col"]], cfg["injection_values"])
        ]
        log.info("%s: %d injection examples found out of %d total.", hf_name, len(injection_rows), len(dataset))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for row in injection_rows:
            text: str = str(row[cfg["text_col"]]).strip()
            if not text:
                continue
            ids.append(_make_doc_id(text))
            documents.append(text)
            metadatas.append({"source": hf_name, "label": "injection"})

        # Batch upsert so a single large dataset doesn't spike memory.
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            collection.upsert(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )
            log.info("  Upserted examples %d–%d.", batch_start, min(batch_end, len(ids)))

        total_upserted += len(ids)
        log.info("Finished %s: %d documents upserted.", hf_name, len(ids))

    log.info(
        "Ingestion complete. %d documents upserted this run. Collection '%s' now has %d total.",
        total_upserted,
        COLLECTION_NAME,
        collection.count(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HuggingFace injection datasets into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before re-ingesting (full rebuild).",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
