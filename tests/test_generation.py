from types import SimpleNamespace

import numpy as np
import pandas as pd

from wharagbot.generation.chat import build_answer_payload, generate_answer
from wharagbot.retrieval.search import DualIndexBundle


def _fake_bundle() -> DualIndexBundle:
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
            }
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
                "style_cluster": 0,
            }
        ]
    )
    fact_units = pd.DataFrame(
        [
            {
                "source_id": "f1",
                "chat_name": "Ana",
                "timestamp": pd.Timestamp("2024-03-01 10:02:00"),
                "fact_text": "Trabajo en remoto como programador",
                "self_fact": True,
                "signal_score": 0.9,
                "context": "Ana: ¿A qué te dedicas?\nWenceslao: Trabajo en remoto como programador",
            }
        ]
    )
    response_embeddings = np.array([[1.0, 0.0]], dtype="float32")
    style_embeddings = np.array([[1.0, 0.0]], dtype="float32")
    fact_embeddings = np.array([[1.0, 0.0]], dtype="float32")
    import faiss

    response_index = faiss.IndexFlatIP(2)
    response_index.add(response_embeddings)
    style_index = faiss.IndexFlatIP(2)
    style_index.add(style_embeddings)
    fact_index = faiss.IndexFlatIP(2)
    fact_index.add(fact_embeddings)
    return DualIndexBundle(
        response_units=response_units,
        style_units=style_units,
        response_index=response_index,
        style_index=style_index,
        response_embeddings=response_embeddings,
        style_embeddings=style_embeddings,
        fact_units=fact_units,
        fact_index=fact_index,
        fact_embeddings=fact_embeddings,
        embedder=None,
    )


def _fake_query_encoder(texts):
    return np.array([[1.0, 0.0]], dtype="float32")


def test_build_answer_payload_includes_style_and_response_hits():
    system_prompt, user_prompt = build_answer_payload(
        my_name="Wenceslao",
        prompt="¿Vamos hoy?",
        retrieval_bundle={
            "fact_hits": [
                {
                    "score": 0.95,
                    "chat_name": "Ana",
                    "timestamp": pd.Timestamp("2024-03-01 10:02:00"),
                    "fact_text": "Trabajo en remoto como programador",
                    "context": "Ana: ¿A qué te dedicas?",
                    "self_fact": True,
                }
            ],
            "response_hits": [
                {
                    "score": 0.9,
                    "chat_name": "Ana",
                    "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
                    "partner_text": "¿Vamos al cine?",
                    "response": "Siiii",
                    "context": "Ana: ¿Vamos al cine?",
                }
            ],
            "style_hits": [
                {
                    "retrieval_role": "style",
                    "style_source": "micro",
                    "style_signature": "short|q0|e0|normal",
                    "style_text": "valeee jaja",
                }
            ],
        },
    )

    assert "Eres Wenceslao escribiendo por WhatsApp" in system_prompt
    assert "Trabajo en remoto como programador" in user_prompt
    assert "Mi respuesta real: Siiii" in user_prompt
    assert "valeee jaja" in user_prompt


def test_build_answer_payload_includes_detected_company_names():
    _, user_prompt = build_answer_payload(
        my_name="Wenceslao",
        prompt="Nombra las empresas donde has trabajado",
        retrieval_bundle={
            "fact_hits": [
                {
                    "score": 0.96,
                    "chat_name": "Ana",
                    "timestamp": pd.Timestamp("2024-03-01 10:02:00"),
                    "fact_text": "Trabajo en Borneo y antes trabajaba en Mar",
                    "context": "Ana: ¿Dónde has trabajado?",
                    "self_fact": True,
                }
            ],
            "response_hits": [],
            "style_hits": [],
        },
    )

    assert "Nombres de empresa detectados en la evidencia" in user_prompt
    assert "- Borneo" in user_prompt
    assert "- Mar" in user_prompt


def test_build_answer_payload_includes_profile_hits():
    _, user_prompt = build_answer_payload(
        my_name="Wenceslao",
        prompt="Como se llama tu pareja actual?",
        retrieval_bundle={
            "profile_hits": [
                {
                    "category": "relationship",
                    "attribute": "partner_current_name",
                    "status": "current",
                    "source_kind": "summary",
                    "rank_score": 0.99,
                    "value": "Adri",
                    "evidence_text": "Mi churri es Adri",
                    "context": "Wenceslao: Mi churri es Adri",
                }
            ],
            "fact_hits": [],
            "response_hits": [],
            "style_hits": [],
        },
    )

    assert "Perfil estructurado sobre mi" in user_prompt
    assert "partner_current_name" in user_prompt
    assert "Valor: Adri" in user_prompt


def test_generate_answer_uses_retry_when_model_echoes_query():
    bundle = _fake_bundle()
    calls = []

    def fake_chat_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            text = "¿Vamos hoy?"
        else:
            text = "Siiii valeee jaja"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    result = generate_answer(
        bundle=bundle,
        my_name="Wenceslao",
        prompt="¿Vamos hoy?",
        api_key="fake",
        gen_model="fake-model",
        k_total=6,
        min_score=0.1,
        chat_create_fn=fake_chat_create,
        encode_queries_fn=_fake_query_encoder,
    )

    assert result["answer"] == "Siiii valeee jaja"
    assert len(calls) == 2
    assert "Reformula sin repetir la consulta." in calls[1]["messages"][1]["content"]
