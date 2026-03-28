import numpy as np
import pandas as pd
import pytest

from wharagbot.retrieval.indexing import IndexBuildSettings, build_dual_indices
from wharagbot.retrieval.search import (
    DualIndexBundle,
    load_dual_index_bundle,
    retrieve_bundle,
)


def _fake_encode(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        low = str(text).lower()
        vector = np.array(
            [
                2.0 if "cine" in low else 0.0,
                2.0 if "comemos" in low else 0.0,
                1.5 if "vale" in low else 0.0,
                1.5 if "jaja" in low else 0.0,
                2.0 if ("dedicas" in low or "program" in low or "remoto" in low) else 0.0,
                2.0 if ("empresa" in low or "borneo" in low or "mar " in low) else 0.0,
                max(1.0, len(low.split()) / 8.0),
            ],
            dtype="float32",
        )
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    return np.stack(vectors).astype("float32")


def test_build_dual_indices_and_retrieve(tmp_path):
    response_units = pd.DataFrame(
        [
            {
                "source_id": "r1",
                "chat_name": "Ana",
                "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
                "partner_text": "¿Vamos al cine?",
                "response": "Siiii",
                "context": "Ana: ¿Vamos al cine?",
                "signal_score": 0.8,
                "reply_gap_min": 1.0,
                "response_embed_text": (
                    "Mensaje del contacto:\n¿Vamos al cine?\n\nContexto:\n"
                    "Ana: ¿Vamos al cine?\n\nMi respuesta real:\nSiiii"
                ),
            },
            {
                "source_id": "r2",
                "chat_name": "Luis",
                "timestamp": pd.Timestamp("2024-03-02 12:00:00"),
                "partner_text": "¿Comemos?",
                "response": "No puedo",
                "context": "Luis: ¿Comemos?",
                "signal_score": 0.7,
                "reply_gap_min": 3.0,
                "response_embed_text": (
                    "Mensaje del contacto:\n¿Comemos?\n\nContexto:\n"
                    "Luis: ¿Comemos?\n\nMi respuesta real:\nNo puedo"
                ),
            },
        ]
    )
    style_units = pd.DataFrame(
        [
            {
                "source_id": "s1",
                "chat_name": "Ana",
                "timestamp": pd.Timestamp("2024-03-01 10:03:00"),
                "style_text": "valeee jaja",
                "style_signature": "short|q0|e0|normal",
                "style_source": "micro",
                "signal_score": 0.5,
                "context": "Ana: ¿Vamos al cine?\nWenceslao: valeee jaja",
                "partner_text": "¿Vamos al cine?",
                "style_embed_text": (
                    "Mi forma habitual de escribir mensajes cortos en "
                    "WhatsApp.\n\nMensaje mio:\nvaleee jaja\n\n"
                    "Rasgos de estilo:\nshort|q0|e0|normal|laugh\n\n"
                    "Ultimo mensaje del contacto:\n¿Vamos al cine?\n\n"
                    "Contexto breve:\nAna: ¿Vamos al cine?"
                ),
            },
            {
                "source_id": "s2",
                "chat_name": "Luis",
                "timestamp": pd.Timestamp("2024-03-02 12:03:00"),
                "style_text": "ok",
                "style_signature": "tiny|q0|e0|normal",
                "style_source": "micro",
                "signal_score": 0.2,
                "context": "Luis: ¿Comemos?\nWenceslao: ok",
                "partner_text": "¿Comemos?",
                "style_embed_text": (
                    "Mi forma habitual de escribir mensajes cortos en "
                    "WhatsApp.\n\nMensaje mio:\nok\n\nRasgos de estilo:\n"
                    "tiny|q0|e0|normal\n\nUltimo mensaje del contacto:\n"
                    "¿Comemos?\n\nContexto breve:\nLuis: ¿Comemos?"
                ),
            },
        ]
    )
    fact_units = pd.DataFrame(
        [
            {
                "source_id": "f1",
                "chat_name": "Ana",
                "timestamp": pd.Timestamp("2024-03-01 10:03:00"),
                "fact_text": "Trabajo en remoto como programador",
                "context": "Ana: ¿A qué te dedicas?\nWenceslao: Trabajo en remoto como programador",
                "partner_text": "¿A qué te dedicas?",
                "signal_score": 0.9,
                "self_fact": True,
                "fact_embed_text": (
                    "Dato sobre mi extraido de mis mensajes de WhatsApp.\n\n"
                    "Tipo: self_fact\n\n"
                    "Mensaje mio:\nTrabajo en remoto como programador\n\n"
                    "Ultimo mensaje del contacto:\n¿A qué te dedicas?\n\n"
                    "Contexto breve:\nAna: ¿A qué te dedicas?"
                ),
            },
            {
                "source_id": "f2",
                "chat_name": "Ana",
                "timestamp": pd.Timestamp("2024-03-04 10:03:00"),
                "fact_text": "Ahora trabajo en Borneo y antes trabajaba en Mar",
                "context": (
                    "Ana: ¿Dónde has trabajado?\n"
                    "Wenceslao: Ahora trabajo en Borneo y antes trabajaba en Mar"
                ),
                "partner_text": "¿Dónde has trabajado?",
                "signal_score": 1.0,
                "self_fact": True,
                "fact_embed_text": (
                    "Dato sobre mi extraido de mis mensajes de WhatsApp.\n\n"
                    "Tipo: self_fact\n\n"
                    "Mensaje mio:\nAhora trabajo en Borneo y antes trabajaba en Mar\n\n"
                    "Ultimo mensaje del contacto:\n¿Dónde has trabajado?\n\n"
                    "Contexto breve:\nAna: ¿Dónde has trabajado?"
                ),
            },
        ]
    )

    result = build_dual_indices(
        response_units,
        style_units,
        fact_units,
        index_dir=tmp_path / "index",
        embed_model="fake-model",
        settings=IndexBuildSettings(),
        encode_passages_fn=_fake_encode,
    )
    bundle = DualIndexBundle(
        response_units=result["response_units"],
        style_units=result["style_units"],
        response_index=result["response_index"],
        style_index=result["style_index"],
        response_embeddings=result["response_embeddings"],
        style_embeddings=result["style_embeddings"],
        fact_units=result["fact_units"],
        fact_index=result["fact_index"],
        fact_embeddings=result["fact_embeddings"],
        embedder=None,
    )

    hits = retrieve_bundle(
        bundle,
        query="¿Vamos al cine esta tarde?",
        k_response=2,
        k_style=3,
        min_score=0.1,
        encode_queries_fn=_fake_encode,
    )

    assert hits["response_hits"]
    assert hits["response_hits"][0]["chat_name"] == "Ana"
    assert hits["style_hits"]
    assert any(
        "vale" in item["style_text"] for item in hits["style_hits"]
    )
    fact_hits = retrieve_bundle(
        bundle,
        query="¿A qué te dedicas?",
        k_response=2,
        k_style=3,
        k_fact=2,
        min_score=0.1,
        encode_queries_fn=_fake_encode,
    )["fact_hits"]
    assert fact_hits
    assert "programador" in fact_hits[0]["fact_text"]

    company_hits = retrieve_bundle(
        bundle,
        query="Nombra las empresas donde has trabajado",
        k_response=2,
        k_style=3,
        k_fact=3,
        min_score=0.1,
        encode_queries_fn=_fake_encode,
    )["fact_hits"]
    assert company_hits
    assert company_hits[0]["company_mentions"] == ["Borneo", "Mar"]


def test_load_dual_index_bundle_requires_existing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="wharagbot build-rag"):
        load_dual_index_bundle(
            index_dir=tmp_path / "index",
            embed_model="fake-model",
            embedder=object(),
        )
