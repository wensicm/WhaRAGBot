from __future__ import annotations

import re
import unicodedata

import pandas as pd


PROFILE_QUERY_STOPWORDS = {
    "a",
    "actual",
    "ahora",
    "como",
    "con",
    "cual",
    "cuál",
    "de",
    "del",
    "donde",
    "el",
    "es",
    "la",
    "las",
    "los",
    "mi",
    "nombre",
    "que",
    "se",
    "sus",
    "tu",
}


def _norm(text: str) -> str:
    text = str(text or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"""[¿?¡!.,;:\-_"'()\[\]]""", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def profile_query_terms(query: str) -> list[str]:
    normalized = _norm(query)
    terms = []
    for token in normalized.split():
        if len(token) < 3 or token in PROFILE_QUERY_STOPWORDS:
            continue
        terms.append(token)

    if any(term in normalized for term in ("pareja", "churri", "novio", "novia")):
        terms.extend(["partner", "pareja", "churri", "novio", "novia"])
    if any(term in normalized for term in ("amigo", "amiga", "hermana")):
        terms.extend(["friend", "best_friend", "sister_like"])
    if any(term in normalized for term in ("comida", "favorita", "gusto")):
        terms.extend(["favorite_food", "likes", "prefers"])
    if "color" in normalized:
        terms.extend(["favorite_color", "likes", "prefers"])
    if any(term in normalized for term in ("trabajo", "empresa", "dedicas")):
        terms.extend(
            [
                "work",
                "company",
                "job_role",
                "company_name",
                "current_company_name",
                "function",
                "schedule",
            ]
        )
    if any(term in normalized for term in ("animal", "gato", "perro", "mascota")):
        terms.extend(["pet", "pet_name", "pet_type"])
    if any(term in normalized for term in ("hablas", "muletilla", "frase")):
        terms.extend(["style", "phrase", "opener", "laugh_marker"])

    seen = set()
    out = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def retrieve_profile_facts(
    profile_facts: pd.DataFrame | None,
    *,
    query: str,
    k: int = 10,
) -> list[dict[str, object]]:
    if profile_facts is None or profile_facts.empty:
        return []

    query_norm = _norm(query)
    terms = profile_query_terms(query)
    wants_current = any(term in query_norm for term in ("actual", "ahora"))
    ranked = []

    for row in profile_facts.to_dict("records"):
        haystack = " ".join(
            [
                str(row.get("category", "")),
                str(row.get("attribute", "")),
                str(row.get("value", "")),
                str(row.get("status", "")),
                str(row.get("evidence_text", "")),
            ]
        )
        haystack_norm = _norm(haystack)
        score = 0.0
        matched = 0
        for term in terms:
            if term and term in haystack_norm:
                matched += 1
                score += 0.08 if len(term) >= 6 else 0.05

        if "partner_current_name" == row.get("attribute") and (
            "pareja" in query_norm or "churri" in query_norm
        ):
            score += 0.30
        if "partner_status_current" == row.get("attribute") and "pareja" in query_norm:
            score += 0.18
        if row.get("category") == "relationship" and "churri" in query_norm:
            score += 0.10
        if row.get("category") == "work" and (
            "trabajo" in query_norm or "empresa" in query_norm
        ):
            score += 0.08
        if row.get("category") == "preference" and (
            "favorit" in query_norm or "gusto" in query_norm
        ):
            score += 0.08
        if row.get("category") == "pet" and (
            "animal" in query_norm or "mascota" in query_norm
        ):
            score += 0.08
        if row.get("category") == "style" and (
            "hablas" in query_norm or "muletilla" in query_norm
        ):
            score += 0.10

        if wants_current and str(row.get("status", "")) == "current":
            score += 0.12
        if str(row.get("source_kind", "")) == "summary":
            score += 0.16

        score += min(0.12, float(row.get("confidence", 0.0)) * 0.12)
        if matched == 0 and score < 0.16:
            continue

        item = dict(row)
        item["rank_score"] = score
        item["retrieval_role"] = "profile"
        ranked.append(item)

    ranked = sorted(ranked, key=lambda item: item["rank_score"], reverse=True)
    out = []
    seen = set()
    for item in ranked:
        key = (
            str(item.get("category", "")),
            str(item.get("attribute", "")),
            str(item.get("value_key", "")),
            str(item.get("source_kind", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= k:
            break
    return out
