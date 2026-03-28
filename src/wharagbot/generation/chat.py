from __future__ import annotations

from typing import Callable

import pandas as pd

from wharagbot.generation.prompts import build_system_prompt
from wharagbot.profile.search import retrieve_profile_facts
from wharagbot.retrieval.search import (
    DualIndexBundle,
    extract_company_mentions,
    norm_text,
    query_requests_company_names,
    retrieve_bundle,
)


def openai_chat_create(
    *,
    api_key: str,
    gen_model: str,
    messages,
    temperature: float | None = None,
):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": gen_model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if (
            "temperature" in message
            and ("unsupported" in message or "only the default" in message)
        ):
            return client.chat.completions.create(
                model=gen_model,
                messages=messages,
            )
        raise


def build_answer_payload(
    *,
    my_name: str,
    prompt: str,
    retrieval_bundle: dict[str, list[dict[str, object]]],
) -> tuple[str, str]:
    profile_hits = retrieval_bundle.get("profile_hits", [])
    response_hits = retrieval_bundle["response_hits"]
    style_hits = retrieval_bundle["style_hits"]
    fact_hits = retrieval_bundle.get("fact_hits", [])
    wants_company_names = query_requests_company_names(prompt)

    company_mentions: list[str] = []
    seen_company_mentions: set[str] = set()
    if wants_company_names:
        for hit in fact_hits:
            mentions = hit.get("company_mentions") or extract_company_mentions(
                str(hit.get("fact_text", hit.get("response", "")) or "")
            )
            for mention in mentions:
                if mention not in seen_company_mentions:
                    seen_company_mentions.add(mention)
                    company_mentions.append(mention)

    response_block = "\n\n".join(
        (
            f"[response | score {hit['score']:.3f} | "
            f"chat: {hit['chat_name']} | fecha: {hit['timestamp']}]\n"
            f"Mensaje del contacto: {hit.get('partner_text', '')}\n"
            f"Mi respuesta real: {hit.get('response', '')}\n"
            f"Contexto: {hit.get('context', '')}"
        )
        for hit in response_hits
    )
    fact_block = "\n\n".join(
        (
            f"[fact | score {hit['score']:.3f} | "
            f"self_fact: {bool(hit.get('self_fact', False))} | "
            f"chat: {hit['chat_name']} | fecha: {hit['timestamp']}]\n"
            f"Mensaje mio: {hit.get('fact_text', hit.get('response', ''))}\n"
            f"Nombres detectados: {', '.join(hit.get('company_mentions', []))}\n"
            f"Contexto: {hit.get('context', '')}"
        )
        for hit in fact_hits
    )
    profile_block = "\n\n".join(
        (
            f"[profile | {hit.get('category', '')}/"
            f"{hit.get('attribute', '')} | "
            f"status: {hit.get('status', '')} | "
            f"source: {hit.get('source_kind', '')} | "
            f"score: {hit.get('rank_score', 0.0):.3f}]\n"
            f"Valor: {hit.get('value', '')}\n"
            f"Evidencia: {hit.get('evidence_text', '')}\n"
            f"Contexto: {hit.get('context', '')}"
        )
        for hit in profile_hits
    )
    style_block = "\n".join(
        (
            f"- [{hit.get('retrieval_role', 'style')} | "
            f"{hit.get('style_source', 'memory')} | "
            f"{hit.get('style_signature', '')}] "
            f"{hit.get('style_text', '')}"
        )
        for hit in style_hits[:10]
    )

    system_prompt = build_system_prompt(my_name)
    user_msg = (
        "Consulta nueva:\n"
        + prompt
        + "\n\nPerfil estructurado sobre mi:\n"
        + (profile_block or "(sin perfil estructurado)")
        + (
            "\n\nNombres de empresa detectados en la evidencia:\n- "
            + "\n- ".join(company_mentions)
            if company_mentions
            else ""
        )
        + "\n\nEvidencia factual sobre mi:\n"
        + (fact_block or "(sin evidencia factual)")
        + "\n\nEvidencia de como respondo a mensajes parecidos:\n"
        + (response_block or "(sin evidencia de respuesta)")
        + "\n\nMuestras de mi estilo de escritura:\n"
        + (style_block or "(sin muestras de estilo)")
        + "\n\nInstrucciones de salida:\n"
        "1) Genera una sola respuesta como la escribiria yo.\n"
        "2) Si la consulta pide un dato sobre mi, usa primero el perfil "
        "estructurado y despues la evidencia factual.\n"
        "3) Si perfil y evidencia factual discrepan, prioriza la mas reciente "
        "y la mas especifica.\n"
        "4) Usa la evidencia de respuesta para completar el contenido solo "
        "si es compatible con los hechos.\n"
        "5) Usa las muestras de estilo para longitud, puntuacion, "
        "muletillas y ritmo, sin copiar literal.\n"
        "6) No suenes como un asistente; suena como un mensaje real mio "
        "de WhatsApp.\n"
        "7) Si falta evidencia para afirmar algo, dilo con claridad.\n"
        "8) No inventes nombres de empresas, personas, ex parejas o sitios.\n"
        "9) Si piden nombres de empresa y hay nombres detectados, limítate "
        "a esos nombres y dilo en una frase corta."
    )
    return system_prompt, user_msg


def _extract_completion_text(completion) -> str:
    return (completion.choices[0].message.content or "").strip()


def generate_answer(
    *,
    bundle: DualIndexBundle,
    my_name: str,
    prompt: str,
    api_key: str,
    gen_model: str,
    k_total: int = 10,
    temperature: float | None = None,
    min_score: float = 0.22,
    profile_facts: pd.DataFrame | None = None,
    chat_create_fn: Callable | None = None,
    encode_queries_fn=None,
) -> dict[str, object]:
    k_response = max(4, k_total // 2)
    k_style = max(6, k_total)
    retrieval_result = retrieve_bundle(
        bundle,
        query=prompt,
        k_response=k_response,
        k_style=k_style,
        min_score=min_score,
        encode_queries_fn=encode_queries_fn,
    )
    retrieval_result["profile_hits"] = retrieve_profile_facts(
        profile_facts,
        query=prompt,
        k=max(8, k_total),
    )
    response_hits = retrieval_result["response_hits"]
    style_hits = retrieval_result["style_hits"]
    fact_hits = retrieval_result.get("fact_hits", [])
    profile_hits = retrieval_result.get("profile_hits", [])

    if not response_hits and not style_hits and not fact_hits and not profile_hits:
        return {
            "answer": (
                "No tengo evidencia suficiente en mis chats "
                "para responder eso con seguridad."
            ),
            "response_hits": response_hits,
            "style_hits": style_hits,
            "fact_hits": fact_hits,
            "profile_hits": profile_hits,
            "system_prompt": build_system_prompt(my_name),
            "user_prompt": "",
        }

    system_prompt, user_prompt = build_answer_payload(
        my_name=my_name,
        prompt=prompt,
        retrieval_bundle=retrieval_result,
    )
    chat_create_fn = chat_create_fn or openai_chat_create

    completion = chat_create_fn(
        api_key=api_key,
        gen_model=gen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    text = _extract_completion_text(completion)
    if not text:
        text = (
            "No tengo evidencia suficiente en mis chats "
            "para responder eso con seguridad."
        )

    if norm_text(text) == norm_text(prompt):
        retry = chat_create_fn(
            api_key=api_key,
            gen_model=gen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                    + "\n\nReformula sin repetir la consulta.",
                },
            ],
            temperature=temperature,
        )
        retried = _extract_completion_text(retry)
        if retried:
            text = retried

    return {
        "answer": text,
        "response_hits": response_hits,
        "style_hits": style_hits,
        "fact_hits": fact_hits,
        "profile_hits": profile_hits,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
