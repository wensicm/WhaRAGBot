from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from wharagbot.utils import load_json, save_json


MEMORY_BUILD_MANIFEST_NAME = "memory_build_manifest.json"
MEMORY_UNITS_FILE_NAME = "memory_units.parquet"
DUAL_UNITS_MANIFEST_NAME = "dual_units_manifest.json"
RESPONSE_UNITS_FILE_NAME = "response_units.parquet"
STYLE_UNITS_FILE_NAME = "style_units.parquet"
FACT_UNITS_FILE_NAME = "fact_units.parquet"


@dataclass(frozen=True)
class MemoryBuildSettings:
    qa_max_gap_min: int = 30
    qa_require_other: bool = True
    qa_require_question: bool = False
    block_min_signal: float = 0.15
    build_version: str = "v12_fact_units"
    block_stride_small: int = 2
    block_stride_medium: int = 6
    block_stride_large: int = 12
    max_blocks_per_topic: int = 120
    max_blocks_total: int = 250000
    min_response_chars: int = 2
    min_style_chars: int = 1
    max_style_words: int = 80
    style_micro_max_words: int = 12
    style_micro_max_chars: int = 48
    style_context_chars: int = 280
    min_fact_chars: int = 8
    fact_context_chars: int = 420


def _quick_messages_fingerprint(messages_df: pd.DataFrame) -> str:
    cols = ["timestamp", "sender", "text", "chat_name"]
    base = messages_df[cols].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")

    n_rows = len(base)
    if n_rows == 0:
        return "empty"

    ts_min = str(base["timestamp"].min())
    ts_max = str(base["timestamp"].max())
    step = max(1, n_rows // 5000)
    sample = base.iloc[::step].head(5000).fillna("")
    sample_hash = pd.util.hash_pandas_object(sample, index=False)

    digest = hashlib.sha256()
    digest.update(f"{n_rows}|{ts_min}|{ts_max}|{step}".encode("utf-8"))
    digest.update(sample_hash.values.tobytes())
    return digest.hexdigest()


def _ingest_manifest_hash(ingest_manifest_path: Path) -> str:
    if not ingest_manifest_path.exists():
        return "missing"
    try:
        return hashlib.sha256(ingest_manifest_path.read_bytes()).hexdigest()
    except Exception:
        return "unreadable"


def build_memory_key(
    messages_df: pd.DataFrame,
    *,
    ingest_manifest_path: Path,
    ctx_window: int,
    my_name: str,
    settings: MemoryBuildSettings,
) -> tuple[str, dict[str, object]]:
    payload = {
        "version": settings.build_version,
        "ingest_manifest_hash": _ingest_manifest_hash(ingest_manifest_path),
        "messages_fp": _quick_messages_fingerprint(messages_df),
        "qa_max_gap_min": settings.qa_max_gap_min,
        "qa_require_other": settings.qa_require_other,
        "qa_require_question": settings.qa_require_question,
        "ctx_window": ctx_window,
        "block_min_signal": settings.block_min_signal,
        "my_name": my_name,
        "block_stride_small": settings.block_stride_small,
        "block_stride_medium": settings.block_stride_medium,
        "block_stride_large": settings.block_stride_large,
        "max_blocks_per_topic": settings.max_blocks_per_topic,
        "max_blocks_total": settings.max_blocks_total,
        "min_response_chars": settings.min_response_chars,
        "min_style_chars": settings.min_style_chars,
        "max_style_words": settings.max_style_words,
        "style_micro_max_words": settings.style_micro_max_words,
        "style_micro_max_chars": settings.style_micro_max_chars,
        "style_context_chars": settings.style_context_chars,
        "min_fact_chars": settings.min_fact_chars,
        "fact_context_chars": settings.fact_context_chars,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload


def style_signature(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "empty"
    words = text.split()
    n_words = len(words)
    question = int(text.endswith("?"))
    exclaim = int(text.endswith("!"))
    alpha_chars = max(1, sum(char.isalpha() for char in text))
    caps_ratio = sum(char.isupper() for char in text) / alpha_chars

    if n_words <= 2:
        length_bucket = "tiny"
    elif n_words <= 6:
        length_bucket = "short"
    elif n_words <= 14:
        length_bucket = "mid"
    else:
        length_bucket = "long"

    tone_bucket = "caps" if caps_ratio > 0.35 else "normal"
    return f"{length_bucket}|q{question}|e{exclaim}|{tone_bucket}"


def clip_tail(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def style_traits(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "empty"

    low = text.lower()
    tags = [style_signature(text)]
    if any(char in text for char in ("?", "¿")):
        tags.append("question")
    if any(char in text for char in ("!", "¡")):
        tags.append("exclaim")
    if "..." in text:
        tags.append("ellipsis")
    if re.search(r"(jaja|jeje|jj|xd)", low):
        tags.append("laugh")
    if re.search(r"(.)\1{2,}", low):
        tags.append("repeat_chars")
    if re.search(r"[A-ZÁÉÍÓÚÑ]{3,}", text):
        tags.append("caps_run")
    if low.startswith(
        ("ok", "vale", "si", "sí", "no", "buah", "uf", "uff", "pues")
    ):
        tags.append("quick_open")
    return "|".join(dict.fromkeys(tags))


def build_qa_examples(
    messages_df: pd.DataFrame,
    *,
    ctx_window: int,
    settings: MemoryBuildSettings,
) -> pd.DataFrame:
    messages_df = messages_df.sort_values("timestamp")
    samples: list[dict[str, object]] = []

    for chat_name, group in messages_df.groupby("chat_name"):
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            if not bool(row.get("is_me", False)):
                continue
            if bool(row.get("is_low_signal", False)):
                continue

            start = max(0, index - ctx_window)
            context_rows = group.iloc[start:index]
            if context_rows.empty:
                continue

            last_other = None
            for candidate in reversed(list(context_rows.itertuples())):
                if not bool(getattr(candidate, "is_me", False)):
                    last_other = candidate
                    break

            if settings.qa_require_other and last_other is None:
                continue

            if last_other is not None:
                delta = row.timestamp - last_other.timestamp
                if pd.notna(delta) and delta > timedelta(
                    minutes=settings.qa_max_gap_min
                ):
                    continue
                reply_gap_min = float(delta.total_seconds() / 60.0)
                partner_text = str(last_other.text)
                partner_sender = str(last_other.sender)
            else:
                reply_gap_min = None
                partner_text = ""
                partner_sender = ""

            if settings.qa_require_question:
                recent = context_rows.tail(4)
                has_question = any(
                    getattr(candidate, "is_question", False)
                    and not bool(getattr(candidate, "is_me", False))
                    for candidate in recent.itertuples()
                )
                if not has_question:
                    continue

            context_text = "\n".join(
                f"{candidate.sender}: {candidate.text}"
                for candidate in context_rows.itertuples()
            )
            response = str(row.text or "").strip()
            samples.append(
                {
                    "unit_type": "qa_turn",
                    "chat_name": chat_name,
                    "timestamp": row.timestamp,
                    "context": context_text,
                    "partner_text": partner_text,
                    "partner_sender": partner_sender,
                    "reply_gap_min": reply_gap_min,
                    "response": response,
                    "style_text": response,
                    "style_signature": style_signature(response),
                    "signal_score": float(row.get("signal_score", 0.5)),
                    "self_fact": bool(row.get("self_fact", False)),
                    "embed_text": (
                        f"Ultimo mensaje del contacto:\n{partner_text}\n\n"
                        f"Contexto reciente:\n{context_text}\n\n"
                        f"Mi respuesta real:\n{response}"
                    ),
                    "source_id": f"qa::{chat_name}::{index}",
                }
            )

    return pd.DataFrame(samples)


def split_topics(
    group: pd.DataFrame,
    *,
    gap_minutes: int = 45,
) -> list[pd.DataFrame]:
    group = group.sort_values("timestamp").reset_index(drop=True)
    topics: list[pd.DataFrame] = []
    current = [0]

    for index in range(1, len(group)):
        delta = group.loc[index, "timestamp"] - group.loc[index - 1, "timestamp"]
        if pd.isna(delta) or delta > timedelta(minutes=gap_minutes):
            topics.append(group.iloc[current].copy())
            current = [index]
        else:
            current.append(index)

    if current:
        topics.append(group.iloc[current].copy())
    return topics


def build_my_message_units(
    messages_df: pd.DataFrame,
    *,
    local_window: int = 2,
) -> pd.DataFrame:
    units: list[dict[str, object]] = []

    for chat_name, group in messages_df.groupby("chat_name"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        prev_other_text = [""] * len(group)
        prev_other_sender = [""] * len(group)
        last_other_text = ""
        last_other_sender = ""
        for index, row in enumerate(group.itertuples()):
            prev_other_text[index] = last_other_text
            prev_other_sender[index] = last_other_sender
            if not bool(getattr(row, "is_me", False)):
                last_other_text = str(row.text)
                last_other_sender = str(row.sender)

        for index, row in group.iterrows():
            if not bool(row.get("is_me", False)):
                continue
            if bool(row.get("is_low_signal", False)):
                continue

            left = max(0, index - local_window)
            right = min(len(group), index + local_window + 1)
            local = group.iloc[left:right]
            local_text = "\n".join(
                f"{candidate.sender}: {candidate.text}"
                for candidate in local.itertuples()
            )

            response = str(row.text or "").strip()
            units.append(
                {
                    "unit_type": "my_message",
                    "chat_name": chat_name,
                    "timestamp": row.timestamp,
                    "context": local_text,
                    "partner_text": prev_other_text[index],
                    "partner_sender": prev_other_sender[index],
                    "reply_gap_min": None,
                    "response": response,
                    "style_text": response,
                    "style_signature": style_signature(response),
                    "signal_score": float(row.get("signal_score", 0.5)),
                    "self_fact": bool(row.get("self_fact", False)),
                    "embed_text": (
                        "Ultimo mensaje del contacto:\n"
                        f"{prev_other_text[index]}\n\n"
                        f"Mensaje mio:\n{response}\n\n"
                        f"Micro-contexto:\n{local_text}"
                    ),
                    "source_id": f"msg::{chat_name}::{index}",
                }
            )

    return pd.DataFrame(units)


def build_topic_blocks(
    messages_df: pd.DataFrame,
    *,
    settings: MemoryBuildSettings,
    min_block: int = 3,
    max_block: int = 8,
    gap_minutes: int = 45,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_blocks = 0

    for chat_name, group in messages_df.groupby("chat_name"):
        topics = split_topics(group, gap_minutes=gap_minutes)
        for topic_index, topic in enumerate(topics):
            topic = topic.reset_index(drop=True)
            n_rows = len(topic)
            if n_rows < min_block:
                continue

            if n_rows < 200:
                stride = settings.block_stride_small
            elif n_rows < 2000:
                stride = settings.block_stride_medium
            else:
                stride = settings.block_stride_large

            per_topic = 0
            for start in range(0, n_rows, stride):
                for size in (min_block, 5, max_block):
                    if size < min_block or size > max_block:
                        continue
                    end = start + size
                    if end > n_rows:
                        continue

                    block = topic.iloc[start:end]
                    block_signal = float(
                        block.get(
                            "signal_score",
                            pd.Series([0.5] * len(block)),
                        ).mean()
                    )
                    if block_signal < settings.block_min_signal:
                        continue

                    text = "\n".join(
                        f"{candidate.sender}: {candidate.text}"
                        for candidate in block.itertuples()
                    )
                    rows.append(
                        {
                            "unit_type": "topic_block",
                            "chat_name": chat_name,
                            "timestamp": block.iloc[-1].timestamp,
                            "context": text,
                            "partner_text": "",
                            "partner_sender": "",
                            "reply_gap_min": None,
                            "response": "",
                            "style_text": "",
                            "style_signature": "block",
                            "signal_score": block_signal,
                            "self_fact": bool(
                                block.get(
                                    "self_fact",
                                    pd.Series([False] * len(block)),
                                ).any()
                            ),
                            "embed_text": f"Bloque de conversacion:\n{text}",
                            "source_id": (
                                f"blk::{chat_name}::{topic_index}::"
                                f"{start}::{end}"
                            ),
                        }
                    )
                    per_topic += 1
                    total_blocks += 1
                    if per_topic >= settings.max_blocks_per_topic:
                        break
                    if total_blocks >= settings.max_blocks_total:
                        break

                if (
                    per_topic >= settings.max_blocks_per_topic
                    or total_blocks >= settings.max_blocks_total
                ):
                    break

            if total_blocks >= settings.max_blocks_total:
                break

        if total_blocks >= settings.max_blocks_total:
            break

    return pd.DataFrame(rows)


def build_response_units(
    memory_units_df: pd.DataFrame,
    *,
    settings: MemoryBuildSettings,
) -> pd.DataFrame:
    base = memory_units_df.copy()
    base = base[base["unit_type"].eq("qa_turn")].copy()
    base["partner_text"] = (
        base["partner_text"].fillna("").astype(str).str.strip()
    )
    base["response"] = base["response"].fillna("").astype(str).str.strip()

    base = base[
        (base["partner_text"].str.len() >= 1)
        & (base["response"].str.len() >= settings.min_response_chars)
    ].copy()

    base["response_embed_text"] = (
        "Mensaje del contacto:\n"
        + base["partner_text"]
        + "\n\nContexto:\n"
        + base["context"].fillna("").astype(str)
        + "\n\nMi respuesta real:\n"
        + base["response"]
    )
    base["retrieval_role"] = "response"
    return base.reset_index(drop=True)


def build_style_micro_units(
    messages_df: pd.DataFrame,
    *,
    settings: MemoryBuildSettings,
    local_window: int = 1,
) -> pd.DataFrame:
    units: list[dict[str, object]] = []

    for chat_name, group in messages_df.groupby("chat_name"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        prev_other_text = [""] * len(group)
        last_other_text = ""
        for index, row in enumerate(group.itertuples()):
            prev_other_text[index] = last_other_text
            if not bool(getattr(row, "is_me", False)):
                last_other_text = str(row.text)

        for index, row in group.iterrows():
            if not bool(row.get("is_me", False)):
                continue

            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            if bool(row.get("has_url", False)):
                continue

            n_words = len(text.split())
            if (
                n_words > settings.style_micro_max_words
                and len(text) > settings.style_micro_max_chars
            ):
                continue

            left = max(0, index - local_window)
            right = min(len(group), index + local_window + 1)
            local = group.iloc[left:right]
            local_text = "\n".join(
                f"{candidate.sender}: {candidate.text}"
                for candidate in local.itertuples()
            )

            units.append(
                {
                    "unit_type": "style_message",
                    "chat_name": chat_name,
                    "timestamp": row.timestamp,
                    "context": local_text,
                    "partner_text": prev_other_text[index],
                    "partner_sender": "",
                    "reply_gap_min": None,
                    "response": text,
                    "style_text": text,
                    "style_signature": style_signature(text),
                    "style_traits": style_traits(text),
                    "style_source": "micro",
                    "signal_score": max(
                        0.15,
                        float(row.get("signal_score", 0.3) or 0.3),
                    ),
                    "self_fact": bool(row.get("self_fact", False)),
                    "source_id": f"sty::{chat_name}::{index}",
                }
            )

    return pd.DataFrame(units)


def build_fact_units(
    messages_df: pd.DataFrame,
    *,
    settings: MemoryBuildSettings,
    local_window: int = 2,
) -> pd.DataFrame:
    units: list[dict[str, object]] = []

    for chat_name, group in messages_df.groupby("chat_name"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        prev_other_text = [""] * len(group)
        prev_other_sender = [""] * len(group)
        last_other_text = ""
        last_other_sender = ""
        for index, row in enumerate(group.itertuples()):
            prev_other_text[index] = last_other_text
            prev_other_sender[index] = last_other_sender
            if not bool(getattr(row, "is_me", False)):
                last_other_text = str(row.text)
                last_other_sender = str(row.sender)

        for index, row in group.iterrows():
            if not bool(row.get("is_me", False)):
                continue

            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            if bool(row.get("has_url", False)):
                continue

            is_self_fact = bool(row.get("self_fact", False))
            is_informative = bool(
                is_self_fact
                or (
                    not bool(row.get("is_low_signal", False))
                    and len(text) >= settings.min_fact_chars
                )
            )
            if not is_informative:
                continue

            left = max(0, index - local_window)
            right = min(len(group), index + local_window + 1)
            local = group.iloc[left:right]
            local_text = "\n".join(
                f"{candidate.sender}: {candidate.text}"
                for candidate in local.itertuples()
            )
            local_text = clip_tail(local_text, settings.fact_context_chars)
            fact_kind = "self_fact" if is_self_fact else "informative_message"

            units.append(
                {
                    "unit_type": "fact_message",
                    "chat_name": chat_name,
                    "timestamp": row.timestamp,
                    "context": local_text,
                    "partner_text": prev_other_text[index],
                    "partner_sender": prev_other_sender[index],
                    "response": text,
                    "fact_text": text,
                    "fact_kind": fact_kind,
                    "signal_score": float(row.get("signal_score", 0.5)),
                    "self_fact": is_self_fact,
                    "fact_embed_text": (
                        "Dato sobre mi extraido de mis mensajes de WhatsApp.\n\n"
                        f"Tipo: {fact_kind}\n\n"
                        f"Mensaje mio:\n{text}\n\n"
                        f"Ultimo mensaje del contacto:\n"
                        f"{prev_other_text[index]}\n\n"
                        f"Contexto breve:\n{local_text}"
                    ),
                    "source_id": f"fact::{chat_name}::{index}",
                }
            )

    facts = pd.DataFrame(units)
    if facts.empty:
        return facts

    facts = facts.sort_values(
        ["self_fact", "signal_score", "timestamp"],
        ascending=[False, False, False],
    )
    facts = facts.drop_duplicates(
        subset=["chat_name", "fact_text"],
        keep="first",
    )
    facts = facts.sort_values("timestamp").reset_index(drop=True)
    return facts


def build_style_units(
    memory_units_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    *,
    settings: MemoryBuildSettings,
) -> pd.DataFrame:
    base = memory_units_df.copy()
    base = base[base["unit_type"].isin(["qa_turn", "my_message"])].copy()
    base["style_text"] = base["response"].fillna("").astype(str).str.strip()
    base = base[base["style_text"].str.len() >= settings.min_style_chars].copy()
    base = base[
        base["style_text"].str.split().str.len().clip(lower=0)
        <= settings.max_style_words
    ].copy()

    base["style_signature"] = base["style_text"].map(style_signature)
    base["style_traits"] = base["style_text"].map(style_traits)
    base["style_source"] = "memory"
    base["style_context"] = base["context"].fillna("").astype(str).map(
        lambda value: clip_tail(value, settings.style_context_chars)
    )
    base["style_partner_text"] = (
        base["partner_text"].fillna("").astype(str).map(
            lambda value: clip_tail(value, settings.style_context_chars)
        )
    )
    base["style_embed_text"] = (
        "Mi forma de escribir en WhatsApp.\n\nMensaje mio:\n"
        + base["style_text"]
        + "\n\nRasgos de estilo:\n"
        + base["style_traits"]
        + "\n\nUltimo mensaje del contacto:\n"
        + base["style_partner_text"]
        + "\n\nContexto breve:\n"
        + base["style_context"]
    )

    micro = build_style_micro_units(messages_df, settings=settings)
    if not micro.empty:
        micro["style_partner_text"] = (
            micro["partner_text"].fillna("").astype(str).map(
                lambda value: clip_tail(value, settings.style_context_chars)
            )
        )
        micro["style_context"] = micro["context"].fillna("").astype(str).map(
            lambda value: clip_tail(value, settings.style_context_chars)
        )
        micro["style_embed_text"] = (
            "Mi forma habitual de escribir mensajes cortos en WhatsApp.\n\n"
            "Mensaje mio:\n"
            + micro["style_text"]
            + "\n\nRasgos de estilo:\n"
            + micro["style_traits"]
            + "\n\nUltimo mensaje del contacto:\n"
            + micro["style_partner_text"]
            + "\n\nContexto breve:\n"
            + micro["style_context"]
        )
        keep_cols = sorted(set(base.columns) | set(micro.columns))
        combined = pd.concat(
            [
                base.reindex(columns=keep_cols),
                micro.reindex(columns=keep_cols),
            ],
            ignore_index=True,
        )
    else:
        combined = base

    combined["retrieval_role"] = "style"
    return combined.sort_values("timestamp").reset_index(drop=True)


def validate_messages_for_memory(messages_df: pd.DataFrame) -> None:
    required_columns = {"timestamp", "sender", "text", "chat_name"}
    missing_columns = required_columns - set(messages_df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            "No se puede construir memoria porque faltan columnas "
            f"obligatorias en messages_clean: {missing_display}."
        )

    if messages_df.empty:
        raise ValueError(
            "No hay mensajes limpios para construir memoria. "
            "Revisa la ingesta y que `CHATS_ZIP_DIR` apunte a exportes "
            "reales de WhatsApp."
        )

    timestamps = pd.to_datetime(messages_df["timestamp"], errors="coerce")
    if timestamps.dropna().empty:
        raise ValueError(
            "No se puede construir memoria porque todos los timestamps de "
            "`messages_clean` son nulos o invalidos."
        )


def build_memory_artifacts(
    messages_df: pd.DataFrame,
    *,
    ctx_window: int,
    settings: MemoryBuildSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_messages_for_memory(messages_df)
    qa_units = build_qa_examples(
        messages_df,
        ctx_window=ctx_window,
        settings=settings,
    )
    my_units = build_my_message_units(messages_df, local_window=2)
    block_units = build_topic_blocks(messages_df, settings=settings)
    memory_units = pd.concat(
        [qa_units, my_units, block_units],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)

    response_units = build_response_units(memory_units, settings=settings)
    style_units = build_style_units(
        memory_units,
        messages_df,
        settings=settings,
    )
    fact_units = build_fact_units(messages_df, settings=settings)
    return memory_units, response_units, style_units, fact_units


def build_or_load_memory_artifacts(
    messages_df: pd.DataFrame,
    *,
    data_dir: Path,
    ingest_manifest_path: Path,
    ctx_window: int,
    my_name: str,
    settings: MemoryBuildSettings,
    force: bool = False,
) -> dict[str, object]:
    memory_manifest_path = data_dir / MEMORY_BUILD_MANIFEST_NAME
    memory_units_path = data_dir / MEMORY_UNITS_FILE_NAME
    dual_manifest_path = data_dir / DUAL_UNITS_MANIFEST_NAME
    response_units_path = data_dir / RESPONSE_UNITS_FILE_NAME
    style_units_path = data_dir / STYLE_UNITS_FILE_NAME
    fact_units_path = data_dir / FACT_UNITS_FILE_NAME

    validate_messages_for_memory(messages_df)
    build_key, build_payload = build_memory_key(
        messages_df,
        ingest_manifest_path=ingest_manifest_path,
        ctx_window=ctx_window,
        my_name=my_name,
        settings=settings,
    )

    memory_manifest = load_json(memory_manifest_path, {})
    dual_manifest = load_json(dual_manifest_path, {})
    memory_cache_hit = (
        not force
        and memory_units_path.exists()
        and memory_manifest.get("build_key") == build_key
    )
    dual_cache_hit = (
        not force
        and response_units_path.exists()
        and style_units_path.exists()
        and fact_units_path.exists()
        and dual_manifest.get("build_key") == build_key
    )

    if memory_cache_hit:
        memory_units = pd.read_parquet(memory_units_path)
    else:
        memory_units, response_units, style_units, fact_units = build_memory_artifacts(
            messages_df,
            ctx_window=ctx_window,
            settings=settings,
        )
        memory_units.to_parquet(memory_units_path, index=False)
        response_units.to_parquet(response_units_path, index=False)
        style_units.to_parquet(style_units_path, index=False)
        fact_units.to_parquet(fact_units_path, index=False)

        save_json(
            memory_manifest_path,
            {
                "build_key": build_key,
                "payload": build_payload,
                "rows": int(len(memory_units)),
                "updated_at": pd.Timestamp.now("UTC").isoformat(),
            },
        )
        save_json(
            dual_manifest_path,
            {
                "build_key": build_key,
                "rows_response": int(len(response_units)),
                "rows_style": int(len(style_units)),
                "rows_fact": int(len(fact_units)),
                "updated_at": pd.Timestamp.now("UTC").isoformat(),
            },
        )
        return {
            "build_key": build_key,
            "build_payload": build_payload,
            "memory_units": memory_units,
            "response_units": response_units,
            "style_units": style_units,
            "fact_units": fact_units,
            "memory_cache_hit": False,
            "dual_cache_hit": False,
        }

    if dual_cache_hit:
        response_units = pd.read_parquet(response_units_path)
        style_units = pd.read_parquet(style_units_path)
        fact_units = pd.read_parquet(fact_units_path)
    else:
        response_units = build_response_units(memory_units, settings=settings)
        style_units = build_style_units(
            memory_units,
            messages_df,
            settings=settings,
        )
        fact_units = build_fact_units(messages_df, settings=settings)
        response_units.to_parquet(response_units_path, index=False)
        style_units.to_parquet(style_units_path, index=False)
        fact_units.to_parquet(fact_units_path, index=False)
        save_json(
            dual_manifest_path,
            {
                "build_key": build_key,
                "rows_response": int(len(response_units)),
                "rows_style": int(len(style_units)),
                "rows_fact": int(len(fact_units)),
                "updated_at": pd.Timestamp.now("UTC").isoformat(),
            },
        )

    return {
        "build_key": build_key,
        "build_payload": build_payload,
        "memory_units": memory_units,
        "response_units": response_units,
        "style_units": style_units,
        "fact_units": fact_units,
        "memory_cache_hit": memory_cache_hit,
        "dual_cache_hit": dual_cache_hit,
    }
