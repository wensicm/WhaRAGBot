from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from wharagbot.embeddings.encoder import build_embedder, encode_passages
from wharagbot.utils import ensure_dir, load_json, save_json


@dataclass(frozen=True)
class IndexBuildSettings:
    response_version: str = "v2_response_e5_query_passage"
    style_version: str = "v3_style_micro_e5_query_passage"
    fact_version: str = "v1_fact_self_message_e5_query_passage"
    style_cluster_version: str = "v1_faiss_kmeans"
    batch_size: int = 32
    verbose: bool = False
    show_progress_bar: bool = False


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def row_hash(row, *, text_col: str, version: str) -> str:
    payload = (
        f"{version}\n<SEP>\n"
        f"{row.get('source_id', '')}\n<SEP>\n"
        f"{row.get('chat_name', '')}\n<SEP>\n"
        f"{row.get('timestamp', '')}\n<SEP>\n"
        f"{row.get(text_col, '')}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embeddings(
    units_df: pd.DataFrame,
    *,
    text_col: str,
    hash_col: str,
    meta_path: Path,
    emb_path: Path,
    version: str,
    encode_passages_fn,
    log_label: str | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    work = units_df.copy()
    work[hash_col] = work.apply(
        lambda row: row_hash(row, text_col=text_col, version=version),
        axis=1,
    )

    cached_map: dict[str, np.ndarray] = {}
    if meta_path.exists() and emb_path.exists():
        cached_units = pd.read_parquet(meta_path)
        cached_emb = np.load(emb_path)
        if hash_col not in cached_units.columns:
            cached_units[hash_col] = cached_units.apply(
                lambda row: row_hash(row, text_col=text_col, version=version),
                axis=1,
            )
        for unit_hash, emb in zip(cached_units[hash_col].tolist(), cached_emb):
            cached_map[str(unit_hash)] = emb

    hashes = work[hash_col].tolist()
    missing = [unit_hash for unit_hash in hashes if unit_hash not in cached_map]
    cached_count = len(hashes) - len(missing)
    if log_label:
        _log(
            f"[build-index] {log_label}: total={len(hashes)} "
            f"cache={cached_count} nuevos={len(missing)}",
            enabled=verbose,
        )
    if missing:
        mask_new = work[hash_col].isin(missing)
        texts = work.loc[mask_new, text_col].fillna("").astype(str).tolist()
        if log_label:
            _log(
                f"[build-index] {log_label}: codificando embeddings nuevos...",
                enabled=verbose,
            )
        new_emb = encode_passages_fn(texts)
        for unit_hash, emb in zip(work.loc[mask_new, hash_col].tolist(), new_emb):
            cached_map[str(unit_hash)] = emb

    emb_matrix = np.stack([cached_map[str(unit_hash)] for unit_hash in hashes])
    return work, emb_matrix.astype("float32"), {
        "total": len(hashes),
        "cached": cached_count,
        "new": len(missing),
    }


def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def style_cluster_key(
    units_df: pd.DataFrame,
    *,
    cluster_version: str,
) -> str:
    raw = "\n".join(units_df["style_unit_hash"].astype(str).tolist())
    payload = f"{cluster_version}\n{raw}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_style_clusters(
    style_units_df: pd.DataFrame,
    style_embeddings: np.ndarray,
    *,
    manifest_path: Path,
    settings: IndexBuildSettings,
) -> tuple[pd.DataFrame, bool]:
    work = style_units_df.copy()
    cluster_key = style_cluster_key(
        work,
        cluster_version=settings.style_cluster_version,
    )
    cluster_manifest = load_json(manifest_path, {})
    cluster_hit = (
        "style_cluster" in work.columns
        and cluster_manifest.get("cluster_key") == cluster_key
    )
    if cluster_hit:
        return work, True

    n_rows = len(style_embeddings)
    if n_rows < 30:
        work["style_cluster"] = 0
    else:
        k = min(24, max(6, int(math.sqrt(n_rows / 180.0))))
        train_size = min(n_rows, 50000)
        if train_size < n_rows:
            rng = np.random.default_rng(42)
            selected = rng.choice(n_rows, size=train_size, replace=False)
            train_x = style_embeddings[selected]
        else:
            train_x = style_embeddings

        kmeans = faiss.Kmeans(
            d=style_embeddings.shape[1],
            k=k,
            niter=20,
            verbose=False,
            seed=42,
            gpu=False,
        )
        kmeans.train(train_x.astype("float32"))
        _, labels = kmeans.index.search(style_embeddings.astype("float32"), 1)
        work["style_cluster"] = labels.reshape(-1).astype(int)

    save_json(
        manifest_path,
        {
            "cluster_key": cluster_key,
            "rows": int(len(work)),
            "updated_at": pd.Timestamp.now("UTC").isoformat(),
        },
    )
    return work, False


def build_dual_indices(
    response_units: pd.DataFrame,
    style_units: pd.DataFrame,
    fact_units: pd.DataFrame,
    *,
    index_dir: Path,
    embed_model: str,
    settings: IndexBuildSettings | None = None,
    embedder=None,
    encode_passages_fn=None,
) -> dict[str, object]:
    settings = settings or IndexBuildSettings()
    ensure_dir(index_dir)

    if response_units.empty or style_units.empty or fact_units.empty:
        raise ValueError("response_units, style_units o fact_units esta vacio")

    response_index_path = index_dir / "response.index"
    style_index_path = index_dir / "style.index"
    fact_index_path = index_dir / "fact.index"
    response_meta_path = index_dir / "response_units.parquet"
    style_meta_path = index_dir / "style_units.parquet"
    fact_meta_path = index_dir / "fact_units.parquet"
    response_emb_path = index_dir / "response_embeddings.npy"
    style_emb_path = index_dir / "style_embeddings.npy"
    fact_emb_path = index_dir / "fact_embeddings.npy"
    style_cluster_manifest_path = index_dir / "style_cluster_manifest.json"

    if encode_passages_fn is None:
        if embedder is None:
            _log(
                f"[build-index] cargando embedder model={embed_model} "
                f"batch_size={settings.batch_size}",
                enabled=settings.verbose,
            )
            embedder = build_embedder(embed_model)
            _log(
                f"[build-index] embedder listo "
                f"device={getattr(embedder, 'device', 'desconocido')}",
                enabled=settings.verbose,
            )

        def encode_passages_fn(texts):
            return encode_passages(
                embedder,
                texts,
                batch_size=settings.batch_size,
                show_progress_bar=settings.show_progress_bar,
            )

    _log(
        "[build-index] fase=response_embeddings",
        enabled=settings.verbose,
    )
    response_units_work, response_embeddings, response_stats = build_embeddings(
        response_units,
        text_col="response_embed_text",
        hash_col="response_unit_hash",
        meta_path=response_meta_path,
        emb_path=response_emb_path,
        version=settings.response_version,
        encode_passages_fn=encode_passages_fn,
        log_label="response_embeddings",
        verbose=settings.verbose,
    )
    response_index = build_faiss_index(response_embeddings)
    faiss.write_index(response_index, str(response_index_path))
    response_units_work.to_parquet(response_meta_path, index=False)
    np.save(response_emb_path, response_embeddings)
    _log(
        "[build-index] response listo "
        f"dim={response_embeddings.shape[1]} "
        f"rows={response_stats['total']} "
        f"path={response_index_path}",
        enabled=settings.verbose,
    )

    _log(
        "[build-index] fase=style_embeddings",
        enabled=settings.verbose,
    )
    style_units_work, style_embeddings, style_stats = build_embeddings(
        style_units,
        text_col="style_embed_text",
        hash_col="style_unit_hash",
        meta_path=style_meta_path,
        emb_path=style_emb_path,
        version=settings.style_version,
        encode_passages_fn=encode_passages_fn,
        log_label="style_embeddings",
        verbose=settings.verbose,
    )
    _log(
        "[build-index] fase=style_clusters",
        enabled=settings.verbose,
    )
    style_units_work, style_cluster_hit = assign_style_clusters(
        style_units_work,
        style_embeddings,
        manifest_path=style_cluster_manifest_path,
        settings=settings,
    )
    style_index = build_faiss_index(style_embeddings)
    faiss.write_index(style_index, str(style_index_path))
    style_units_work.to_parquet(style_meta_path, index=False)
    np.save(style_emb_path, style_embeddings)
    _log(
        "[build-index] style listo "
        f"dim={style_embeddings.shape[1]} "
        f"rows={style_stats['total']} "
        f"clusters={style_units_work['style_cluster'].nunique()} "
        f"cluster_cache_hit={style_cluster_hit} "
        f"path={style_index_path}",
        enabled=settings.verbose,
    )

    _log(
        "[build-index] fase=fact_embeddings",
        enabled=settings.verbose,
    )
    fact_units_work, fact_embeddings, fact_stats = build_embeddings(
        fact_units,
        text_col="fact_embed_text",
        hash_col="fact_unit_hash",
        meta_path=fact_meta_path,
        emb_path=fact_emb_path,
        version=settings.fact_version,
        encode_passages_fn=encode_passages_fn,
        log_label="fact_embeddings",
        verbose=settings.verbose,
    )
    fact_index = build_faiss_index(fact_embeddings)
    faiss.write_index(fact_index, str(fact_index_path))
    fact_units_work.to_parquet(fact_meta_path, index=False)
    np.save(fact_emb_path, fact_embeddings)
    _log(
        "[build-index] fact listo "
        f"dim={fact_embeddings.shape[1]} "
        f"rows={fact_stats['total']} "
        f"path={fact_index_path}",
        enabled=settings.verbose,
    )

    return {
        "response_units": response_units_work,
        "style_units": style_units_work,
        "fact_units": fact_units_work,
        "response_embeddings": response_embeddings,
        "style_embeddings": style_embeddings,
        "fact_embeddings": fact_embeddings,
        "response_stats": response_stats,
        "style_stats": style_stats,
        "fact_stats": fact_stats,
        "style_cluster_cache_hit": style_cluster_hit,
        "response_index": response_index,
        "style_index": style_index,
        "fact_index": fact_index,
        "response_index_path": response_index_path,
        "style_index_path": style_index_path,
        "fact_index_path": fact_index_path,
        "response_meta_path": response_meta_path,
        "style_meta_path": style_meta_path,
        "fact_meta_path": fact_meta_path,
        "response_emb_path": response_emb_path,
        "style_emb_path": style_emb_path,
        "fact_emb_path": fact_emb_path,
    }
