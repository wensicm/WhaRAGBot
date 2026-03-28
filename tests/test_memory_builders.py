from pathlib import Path

import pandas as pd
import pytest

from wharagbot.memory.builders import (
    MemoryBuildSettings,
    build_memory_artifacts,
    build_or_load_memory_artifacts,
)


def _sample_clean_messages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
                "sender": "Ana",
                "text": "¿Vamos al cine?",
                "chat_name": "Ana",
                "is_me": False,
                "is_low_signal": False,
                "is_question": True,
                "signal_score": 0.8,
                "self_fact": False,
                "has_url": False,
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
                "sender": "Wenceslao",
                "text": "Siiii",
                "chat_name": "Ana",
                "is_me": True,
                "is_low_signal": False,
                "is_question": False,
                "signal_score": 0.5,
                "self_fact": False,
                "has_url": False,
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:02:00"),
                "sender": "Ana",
                "text": "Jajaja dale",
                "chat_name": "Ana",
                "is_me": False,
                "is_low_signal": False,
                "is_question": False,
                "signal_score": 0.4,
                "self_fact": False,
                "has_url": False,
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:03:00"),
                "sender": "Wenceslao",
                "text": "valeee jaja",
                "chat_name": "Ana",
                "is_me": True,
                "is_low_signal": False,
                "is_question": False,
                "signal_score": 0.4,
                "self_fact": False,
                "has_url": False,
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 11:00:00"),
                "sender": "Luis",
                "text": "ok",
                "chat_name": "Luis",
                "is_me": False,
                "is_low_signal": True,
                "is_question": False,
                "signal_score": 0.1,
                "self_fact": False,
                "has_url": False,
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 11:05:00"),
                "sender": "Wenceslao",
                "text": "Trabajo en remoto como programador",
                "chat_name": "Luis",
                "is_me": True,
                "is_low_signal": False,
                "is_question": False,
                "signal_score": 0.9,
                "self_fact": True,
                "has_url": False,
            },
        ]
    )


def test_build_memory_artifacts_creates_dual_units():
    settings = MemoryBuildSettings()
    messages = _sample_clean_messages()

    memory_units, response_units, style_units, fact_units = build_memory_artifacts(
        messages,
        ctx_window=4,
        settings=settings,
    )

    assert set(memory_units["unit_type"]) >= {
        "qa_turn",
        "my_message",
        "topic_block",
    }
    assert set(response_units["unit_type"]) == {"qa_turn"}
    assert "micro" in set(style_units["style_source"])
    assert "memory" in set(style_units["style_source"])
    assert style_units["style_text"].str.contains("valeee").any()
    assert not fact_units.empty
    assert "Trabajo en remoto como programador" in set(fact_units["fact_text"])


def test_build_or_load_memory_artifacts_uses_manifest_cache(tmp_path: Path):
    settings = MemoryBuildSettings()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ingest_manifest_path = data_dir / "ingest_cache" / "manifest.json"
    ingest_manifest_path.parent.mkdir()
    ingest_manifest_path.write_text("{}")

    messages = _sample_clean_messages()
    first = build_or_load_memory_artifacts(
        messages,
        data_dir=data_dir,
        ingest_manifest_path=ingest_manifest_path,
        ctx_window=4,
        my_name="Wenceslao",
        settings=settings,
        force=False,
    )
    second = build_or_load_memory_artifacts(
        messages,
        data_dir=data_dir,
        ingest_manifest_path=ingest_manifest_path,
        ctx_window=4,
        my_name="Wenceslao",
        settings=settings,
        force=False,
    )

    assert first["memory_cache_hit"] is False
    assert second["memory_cache_hit"] is True
    assert second["dual_cache_hit"] is True


def test_build_memory_artifacts_rejects_empty_messages():
    settings = MemoryBuildSettings()

    with pytest.raises(ValueError, match="No hay mensajes limpios"):
        build_memory_artifacts(
            pd.DataFrame(
                columns=["timestamp", "sender", "text", "chat_name"]
            ),
            ctx_window=4,
            settings=settings,
        )
