from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from wharagbot.retrieval.search import (
    extract_company_mentions,
    work_context_score,
    worked_company_evidence_score,
)
from wharagbot.utils import load_json, save_json


PROFILE_FACTS_FILE_NAME = "profile_facts.parquet"
PROFILE_SUMMARY_FILE_NAME = "profile_summary.json"
PROFILE_MANIFEST_NAME = "profile_manifest.json"

NAME_STOPWORDS = {
    "abril",
    "ahora",
    "algo",
    "aunque",
    "bien",
    "borneo",
    "bueno",
    "camarero",
    "carli",
    "chatgpt",
    "chicho",
    "como",
    "con",
    "corte",
    "cristina",
    "david",
    "de",
    "del",
    "desarrollador",
    "donde",
    "en",
    "equipo",
    "era",
    "es",
    "esta",
    "estoy",
    "fuerteventura",
    "google",
    "hoy",
    "ingrid",
    "jamila",
    "jose",
    "kaleta",
    "la",
    "las",
    "lo",
    "los",
    "mad",
    "madrid",
    "makro",
    "manu",
    "mar",
    "mercadona",
    "mi",
    "mira",
    "mery",
    "no",
    "oswaldo",
    "pero",
    "porque",
    "poema",
    "que",
    "restservation",
    "si",
    "sip",
    "terraza",
    "tiktok",
    "true",
    "tu",
    "vips",
    "wenceslao",
    "yo",
}

FRIEND_NAME_STOPWORDS = {
    "amigos",
    "amigas",
    "controladores",
    "importantes",
    "persona",
    "personas",
}

PLACE_PREFIX_STOPWORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "la",
    "las",
    "lo",
    "los",
    "mi",
    "su",
    "tu",
    "un",
    "una",
}

PLACE_REGION_WORDS = {
    "centro",
    "isla",
    "norte",
    "sur",
}

JOB_ROLE_KEYWORDS = (
    "analista",
    "barista",
    "cajer",
    "camarer",
    "chef",
    "cociner",
    "data scientist",
    "dependiente",
    "desarrollador",
    "encargad",
    "hosteler",
    "ingenier",
    "programador",
    "recepcionista",
)

JOB_ROLE_NEGATIVE_HINTS = (
    "buscar un trabajo de",
    "buscando videos de como trabajo de",
    "como trabajo de",
    "luchando por un trabajo de",
    "opcion seria buscar un trabajo de",
    "opción sería buscar un trabajo de",
    "por un trabajo de",
    "si trabajo de",
    "un trabajo de",
)

STYLE_OPENERS = {
    "ahh",
    "aiii",
    "buah",
    "bueno",
    "holi",
    "nah",
    "ok",
    "ostras",
    "pues",
    "sip",
    "sii",
    "vale",
    "vooy",
    "yap",
}

PREFERENCE_PATTERNS = [
    (
        "favorite_food",
        re.compile(r"\bmi\s+comida\s+favorita\s+es\s+(.+)", flags=re.I),
    ),
    (
        "favorite_color",
        re.compile(
            r"\bmi\s+color\s+(?:favorito|preferido)\s+es\s+(.+)",
            flags=re.I,
        ),
    ),
    (
        "favorite_movie",
        re.compile(r"\bmi\s+pel[ií]cula\s+favorita\s+es\s+(.+)", flags=re.I),
    ),
    (
        "favorite_series",
        re.compile(r"\bmi\s+serie\s+favorita\s+es\s+(.+)", flags=re.I),
    ),
    (
        "favorite_song",
        re.compile(r"\bmi\s+canci[oó]n\s+favorita\s+es\s+(.+)", flags=re.I),
    ),
    (
        "favorite_book",
        re.compile(r"\bmi\s+libro\s+favorito\s+es\s+(.+)", flags=re.I),
    ),
]

PET_PATTERN = re.compile(
    r"\bmi\s+(gato|gata|perro|perra|conejo|hamster|hur[oó]n|tortuga|loro)\s+"
    r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+)",
    flags=re.I,
)


@dataclass(frozen=True)
class ProfileBuildSettings:
    build_version: str = "v2_structured_profile_precision"
    local_window: int = 2
    context_chars: int = 420
    style_top_k: int = 8
    style_min_count: int = 3


def _clip_tail(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _normalize_value(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,;:!?()[]{}\"'")
    return text


def _value_key(text: str) -> str:
    return _normalize_value(text).casefold()


def _source_id(kind: str, category: str, attribute: str, value: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", _value_key(value)).strip("-")
    return f"{kind}::{category}::{attribute}::{safe or 'empty'}"


def _append_fact(
    facts: list[dict[str, object]],
    *,
    category: str,
    attribute: str,
    value: str,
    timestamp=None,
    chat_name: str = "",
    evidence_text: str = "",
    context: str = "",
    status: str = "mentioned",
    confidence: float = 0.7,
    source_kind: str = "raw",
    evidence_count: int = 1,
) -> None:
    value = _normalize_value(value)
    if not value:
        return
    facts.append(
        {
            "category": category,
            "attribute": attribute,
            "value": value,
            "value_key": _value_key(value),
            "status": status,
            "confidence": float(confidence),
            "source_kind": source_kind,
            "evidence_count": int(evidence_count),
            "timestamp": timestamp,
            "chat_name": chat_name,
            "evidence_text": _normalize_value(evidence_text),
            "context": context,
            "source_id": _source_id(source_kind, category, attribute, value),
        }
    )


def _local_context(
    group: pd.DataFrame,
    *,
    index: int,
    window: int,
    max_chars: int,
) -> str:
    left = max(0, index - window)
    right = min(len(group), index + window + 1)
    local = group.iloc[left:right]
    text = "\n".join(
        f"{candidate.sender}: {candidate.text}"
        for candidate in local.itertuples()
    )
    return _clip_tail(text, max_chars)


def _split_names(raw: str) -> list[str]:
    text = _normalize_value(raw)
    if not text:
        return []
    parts = re.split(r"\s+y\s+|,", text)
    names = []
    for part in parts:
        candidate = _normalize_value(part)
        if not candidate:
            continue
        if any(
            token.casefold() in FRIEND_NAME_STOPWORDS
            for token in candidate.split()
        ):
            continue
        normalized = _normalize_person_name(candidate)
        if normalized:
            names.append(normalized)
    return names


def _normalize_person_name(raw: str) -> str | None:
    candidate = _normalize_value(raw)
    if not candidate:
        return None
    tokens = candidate.split()
    if len(tokens) > 3:
        return None
    if any(token.casefold() in NAME_STOPWORDS for token in tokens):
        return None
    if not all(token[:1].isupper() for token in tokens):
        return None
    return candidate


def _extract_place_value(raw: str) -> tuple[str | None, float]:
    text = _normalize_value(raw)
    if not text:
        return None, 0.0
    text = re.sub(
        r"^(?:yo|pues|bueno|nah|noo|la\s+verdad\s+es\s+que)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, 0.0

    region_match = re.match(
        r"^(?:(el|la)\s+)?"
        r"(norte|sur|centro|isla)\s+de\s+"
        r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})",
        text,
    )
    if region_match:
        article = (region_match.group(1) or "").strip()
        region = region_match.group(2)
        place = region_match.group(3)
        prefix = f"{article} {region}".strip()
        return f"{prefix} de {place}".strip(), 0.88

    candidate_tokens = []
    connector_tokens = {"de", "del", "la", "las", "los", "el"}
    for token in text.split():
        token = token.strip(".,;:!?()[]{}\"'")
        if not token:
            continue
        is_capitalized = token[:1].isupper()
        is_connector = token.casefold() in connector_tokens

        if not candidate_tokens:
            if is_capitalized:
                candidate_tokens.append(token)
                continue
            break

        if is_capitalized or is_connector:
            candidate_tokens.append(token)
            continue
        break

    while candidate_tokens and candidate_tokens[-1].casefold() in connector_tokens:
        candidate_tokens.pop()
    if not candidate_tokens:
        return None, 0.0

    value = _normalize_value(" ".join(candidate_tokens))
    tokens = value.split()
    if not tokens:
        return None, 0.0
    if tokens[0].casefold() in PLACE_PREFIX_STOPWORDS and len(tokens) == 1:
        return None, 0.0
    if any(token.casefold() in PLACE_REGION_WORDS for token in tokens[1:]):
        return None, 0.0
    confidence = 0.82 if len(tokens) > 1 else 0.5
    return value, confidence


def _normalize_job_role(raw: str) -> str | None:
    value = _normalize_value(raw)
    if not value:
        return None
    value = re.split(
        r"(?i)\s+(?:pero|porque|que|si|cuando|aunque|con|para|y)\s+",
        value,
        maxsplit=1,
    )[0]
    value = re.sub(
        r"(?i)\s+(?:los\s+fines|de\s+noche|por\s+las\s+noches).*$",
        "",
        value,
    )
    value = _normalize_value(value)
    if not value:
        return None
    if re.search(r"\d", value):
        return None
    if len(value.split()) > 4:
        return None
    normalized = value.casefold()
    if not any(keyword in normalized for keyword in JOB_ROLE_KEYWORDS):
        return None
    return value


def _extract_partner_names_from_context(context: str) -> list[str]:
    direct_patterns = [
        re.compile(
            r"(?i:\bquien\s+es\s+)([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
            r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\s+"
            r"(?i:el\s+churri\b)",
        ),
        re.compile(
            r"(?i:\bmi\s+(?:churri|pareja|novi[oa])(?:\s+actual)?\s+"
            r"(?:es|se\s+llama)\s+)"
            r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
            r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})",
        ),
        re.compile(
            r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
            r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\s+"
            r"(?i:(?:es|era)\s+mi\s+(?:churri|pareja|novi[oa])\b)",
        ),
    ]
    mention_pattern = re.compile(
        r"(?i:\bmenciono\s+a\s+)([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\b",
    )
    out = []
    seen = set()
    lines = []
    for raw_line in str(context or "").splitlines():
        line = raw_line.split(":", 1)[-1].strip()
        if line:
            lines.append(line)

    for line in lines:
        for pattern in direct_patterns:
            for match in pattern.finditer(line):
                name = _normalize_person_name(match.group(1))
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)

    has_partner_anchor = any(
        re.search(r"\bmi\s+(?:churri|pareja|novi[oa])\b", line, re.I)
        for line in lines
    )
    if has_partner_anchor:
        for line in lines:
            for match in mention_pattern.finditer(line):
                name = _normalize_person_name(match.group(1))
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def _extract_identity_and_location(
    *,
    facts: list[dict[str, object]],
    text: str,
    context: str,
    timestamp,
    chat_name: str,
    first_name: str,
) -> None:
    full_name_pattern = re.compile(
        rf"\b({re.escape(first_name)}\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){1,2})\b"
    )
    for pattern in [
        re.compile(r"\bme\s+llamo\s+(.+)", flags=re.I),
        re.compile(r"\bmi\s+nombre\s+es\s+(.+)", flags=re.I),
    ]:
        match = pattern.search(text)
        if match:
            _append_fact(
                facts,
                category="identity",
                attribute="full_name",
                value=match.group(1),
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="current",
                confidence=0.95,
            )

    for match in full_name_pattern.finditer(text):
        _append_fact(
            facts,
            category="identity",
            attribute="full_name",
            value=match.group(1),
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.9,
        )

    for attribute, pattern in [
        (
            "lives_in",
            re.compile(
                r"(?<!lo\s)\bvivo\s+en\s+(.+)",
                flags=re.I,
            ),
        ),
        (
            "from_place",
            re.compile(
                r"(?<!no\s)\bsoy\s+de\s+(.+)",
                flags=re.I,
            ),
        ),
    ]:
        match = pattern.search(text)
        if match:
            value, confidence = _extract_place_value(match.group(1))
            if not value:
                continue
            _append_fact(
                facts,
                category="location",
                attribute=attribute,
                value=value,
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="current",
                confidence=confidence,
            )


def _extract_preferences(
    *,
    facts: list[dict[str, object]],
    text: str,
    context: str,
    timestamp,
    chat_name: str,
) -> None:
    for attribute, pattern in PREFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        _append_fact(
            facts,
            category="preference",
            attribute=attribute,
            value=match.group(1),
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.95,
        )

    for attribute, pattern in [
        ("likes", re.compile(r"\bme\s+gusta(?:\s+mucho)?\s+(.+)", flags=re.I)),
        ("prefers", re.compile(r"\bprefiero\s+(.+)", flags=re.I)),
        ("passion", re.compile(r"\bme\s+apasiona\s+(.+)", flags=re.I)),
    ]:
        match = pattern.search(text)
        if not match:
            continue
        value = _normalize_value(match.group(1))
        if len(value.split()) > 8:
            continue
        _append_fact(
            facts,
            category="preference",
            attribute=attribute,
            value=value,
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.72,
        )


def _extract_relationships(
    *,
    facts: list[dict[str, object]],
    text: str,
    context: str,
    timestamp,
    chat_name: str,
) -> None:
    direct_partner = re.compile(
        r"(?i:\bmi\s+(?:churri|pareja|novi[oa])(?:\s+actual)?\s+"
        r"(?:es|se\s+llama)\s+)"
        r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})",
    )
    match = direct_partner.search(text)
    if match:
        _append_fact(
            facts,
            category="relationship",
            attribute="partner_name",
            value=match.group(1),
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.98,
        )

    if re.search(r"\b(?:ahora\s+mismo\s+)?no\s+tengo\s+pareja(?:\s+actual)?\b", text, re.I):
        _append_fact(
            facts,
            category="relationship",
            attribute="partner_status",
            value="single",
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.98,
        )

    if re.search(
        r"\b(?:ya\s+no\s+somos\s+pareja|cuando\s+[ée]ramos\s+pareja|"
        r"[ée]ramos\s+pareja|no\s+como\s+pareja)\b",
        text,
        re.I,
    ):
        _append_fact(
            facts,
            category="relationship",
            attribute="partner_status",
            value="not_together",
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="past",
            confidence=0.9,
        )

    if (
        re.search(r"\b(?:hemos\s+vuelto|desde\s+que\s+hemos\s+vuelto)\b", text, re.I)
        and re.search(
            r"\b(?:pareja|churri|relaci[oó]n|ex|novi[oa])\b",
            text + "\n" + context,
            re.I,
        )
    ):
        _append_fact(
            facts,
            category="relationship",
            attribute="partner_status",
            value="together",
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.82,
        )

    if not match and re.search(r"\bmi\s+(?:churri|pareja|novi[oa])\b", text, re.I):
        for candidate in _extract_partner_names_from_context(context):
            _append_fact(
                facts,
                category="relationship",
                attribute="partner_name",
                value=candidate,
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="candidate",
                confidence=0.84,
            )

    for match in re.finditer(
        r"(?i:\bmi\s+mejor\s+amig[oa]\s+"
        r"(?:es|se\s+llama)\s+)"
        r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})",
        text,
    ):
        name = _normalize_person_name(match.group(1))
        if not name:
            continue
        _append_fact(
            facts,
            category="relationship",
            attribute="best_friend_name",
            value=name,
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.95,
        )

    for match in re.finditer(
        r"\bmis\s+amig(?:os|as)\s+(.+)",
        text,
        flags=re.I,
    ):
        for name in _split_names(match.group(1)):
            _append_fact(
                facts,
                category="relationship",
                attribute="friend_name",
                value=name,
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="current",
                confidence=0.75,
            )

    for pattern in [
        re.compile(
            r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+)\s+es\s+como\s+una\s+hermana\b"
        ),
        re.compile(
            r"\bconsidero\s+(?:a\s+)?([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+)\s+"
            r"(?:como\s+)?una\s+hermana\b",
            flags=re.I,
        ),
    ]:
        for match in pattern.finditer(text):
            _append_fact(
                facts,
                category="relationship",
                attribute="sister_like_name",
                value=match.group(1),
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="current",
                confidence=0.8,
            )


def _extract_work(
    *,
    facts: list[dict[str, object]],
    text: str,
    context: str,
    timestamp,
    chat_name: str,
) -> None:
    work_score = work_context_score(text + "\n" + context)
    mentions = extract_company_mentions(text)
    worked_score = worked_company_evidence_score(text)

    if mentions and worked_score > 0:
        status = "mentioned"
        if re.search(
            r"\b(?:trabajo\s+en|alta\s+en|hace\s+un\s+a[nñ]o\s+que\s+trabajo\s+en|"
            r"trabajando\s+en|ahora\s+trabajo\s+en)\b",
            text,
            re.I,
        ):
            status = "current"
        elif re.search(
            r"\b(?:trabaj[ée]\s+en|he\s+trabajado\s+en|mi\s+antiguo\s+trabajo)\b",
            text,
            re.I,
        ):
            status = "past"

        for company_name in mentions:
            _append_fact(
                facts,
                category="work",
                attribute="company_name",
                value=company_name,
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status=status,
                confidence=0.9 if status != "mentioned" else 0.74,
            )

    normalized_text = text.casefold()
    for attribute, pattern in [
        (
            "job_role",
            re.compile(
                r"^(?:yo\s+|pues\s+|bueno\s+)?trabajo\s+de\s+(.+)$",
                flags=re.I,
            ),
        ),
        (
            "job_role",
            re.compile(
                r"^(?:yo\s+|pues\s+|bueno\s+)?curro\s+de\s+(.+)$",
                flags=re.I,
            ),
        ),
        ("job_role", re.compile(r"^me\s+dedico\s+a\s+(.+)$", flags=re.I)),
        (
            "job_role",
            re.compile(
                r"^(?:yo\s+|pues\s+|bueno\s+)?trabajo\s+en\s+la\s+"
                r"([a-záéíóúñ ]+)$",
                flags=re.I,
            ),
        ),
    ]:
        if any(hint in normalized_text for hint in JOB_ROLE_NEGATIVE_HINTS):
            break
        match = pattern.search(text)
        if not match:
            continue
        value = _normalize_job_role(match.group(1))
        if not value:
            continue
        _append_fact(
            facts,
            category="work",
            attribute=attribute,
            value=value,
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
                context=context,
            status="current",
            confidence=0.9,
        )

    schedule_patterns = [
        re.compile(
            r"\bde\s+\d{1,2}(?::\d{2})?\s*(?:a|-)\s*\d{1,2}(?::\d{2})?\s*h?\b",
            flags=re.I,
        ),
        re.compile(r"\bhasta\s+las\s+\d{1,2}(?::\d{2})?\b", flags=re.I),
        re.compile(r"\ba\s+las\s+\d{1,2}(?::\d{2})?\b", flags=re.I),
    ]
    for pattern in schedule_patterns:
        for match in pattern.finditer(text):
            _append_fact(
                facts,
                category="work",
                attribute="schedule",
                value=match.group(0),
                timestamp=timestamp,
                chat_name=chat_name,
                evidence_text=text,
                context=context,
                status="mentioned",
                confidence=0.75,
            )

    if work_score > 0:
        function_terms = {
            "facturacion": ["factur"],
            "compras": ["compra", "makro", "mercadona"],
            "stock": ["stock"],
            "reservas": ["reserva"],
            "proveedores": ["proveedor"],
            "pedidos": ["pedido", "escandallo"],
            "carta_y_precios": ["carta", "vinos", "precios"],
            "uniformes": ["uniforme"],
        }
        normalized = text.casefold()
        for value, stems in function_terms.items():
            if any(stem in normalized for stem in stems):
                _append_fact(
                    facts,
                    category="work",
                    attribute="function",
                    value=value,
                    timestamp=timestamp,
                    chat_name=chat_name,
                    evidence_text=text,
                    context=context,
                    status="mentioned",
                    confidence=0.72,
                )


def _extract_pets(
    *,
    facts: list[dict[str, object]],
    text: str,
    context: str,
    timestamp,
    chat_name: str,
) -> None:
    for match in PET_PATTERN.finditer(text):
        pet_type = _normalize_value(match.group(1).lower())
        pet_name = _normalize_value(match.group(2))
        _append_fact(
            facts,
            category="pet",
            attribute="pet_name",
            value=pet_name,
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.92,
        )
        _append_fact(
            facts,
            category="pet",
            attribute="pet_type",
            value=pet_type,
            timestamp=timestamp,
            chat_name=chat_name,
            evidence_text=text,
            context=context,
            status="current",
            confidence=0.92,
        )


def _extract_raw_profile_facts(
    messages_df: pd.DataFrame,
    *,
    my_name: str,
    settings: ProfileBuildSettings,
) -> pd.DataFrame:
    facts: list[dict[str, object]] = []
    first_name = _normalize_value(my_name).split()[0] if my_name else ""

    for chat_name, group in messages_df.groupby("chat_name"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for index, row in group.iterrows():
            if not bool(row.get("is_me", False)):
                continue
            text = _normalize_value(row.get("text", ""))
            if not text:
                continue

            context = _local_context(
                group,
                index=index,
                window=settings.local_window,
                max_chars=settings.context_chars,
            )
            _extract_identity_and_location(
                facts=facts,
                text=text,
                context=context,
                timestamp=row.get("timestamp"),
                chat_name=chat_name,
                first_name=first_name,
            )
            _extract_preferences(
                facts=facts,
                text=text,
                context=context,
                timestamp=row.get("timestamp"),
                chat_name=chat_name,
            )
            _extract_relationships(
                facts=facts,
                text=text,
                context=context,
                timestamp=row.get("timestamp"),
                chat_name=chat_name,
            )
            _extract_work(
                facts=facts,
                text=text,
                context=context,
                timestamp=row.get("timestamp"),
                chat_name=chat_name,
            )
            _extract_pets(
                facts=facts,
                text=text,
                context=context,
                timestamp=row.get("timestamp"),
                chat_name=chat_name,
            )

    raw_facts = pd.DataFrame(facts)
    if raw_facts.empty:
        return raw_facts

    raw_facts = raw_facts.sort_values(
        ["timestamp", "confidence"],
        ascending=[False, False],
    )
    raw_facts = raw_facts.drop_duplicates(
        subset=["category", "attribute", "value_key", "status", "source_kind"],
        keep="first",
    )
    return raw_facts.sort_values("timestamp").reset_index(drop=True)


def _make_summary_rows(
    raw_facts: pd.DataFrame,
    *,
    messages_df: pd.DataFrame,
    my_name: str,
    settings: ProfileBuildSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp_max = pd.to_datetime(
        messages_df["timestamp"],
        errors="coerce",
    ).max()

    _append_fact(
        rows,
        category="identity",
        attribute="first_name",
        value=my_name,
        timestamp=timestamp_max,
        status="current",
        confidence=1.0,
        source_kind="summary",
    )

    def latest_fact(category: str, attribute: str) -> pd.Series | None:
        subset = raw_facts[
            (raw_facts["category"] == category)
            & (raw_facts["attribute"] == attribute)
        ].sort_values(["timestamp", "confidence"], ascending=[False, False])
        if subset.empty:
            return None
        return subset.iloc[0]

    for attribute in [
        "full_name",
        "favorite_food",
        "favorite_color",
        "favorite_movie",
        "favorite_series",
        "favorite_song",
        "favorite_book",
    ]:
        category = "identity"
        if attribute in {"lives_in", "from_place"}:
            category = "location"
        elif attribute.startswith("favorite_"):
            category = "preference"
        row = latest_fact(category, attribute)
        if row is None:
            continue
        _append_fact(
            rows,
            category=category,
            attribute=attribute,
            value=row["value"],
            timestamp=row["timestamp"],
            chat_name=row.get("chat_name", ""),
            evidence_text=row.get("evidence_text", ""),
            context=row.get("context", ""),
            status=row.get("status", "current"),
            confidence=min(1.0, float(row["confidence"]) + 0.05),
            source_kind="summary",
        )

    for attribute in ["lives_in", "from_place"]:
        subset = raw_facts[
            (raw_facts["category"] == "location")
            & (raw_facts["attribute"] == attribute)
        ]
        if subset.empty:
            continue
        grouped = (
            subset.groupby("value")
            .agg(
                latest_timestamp=("timestamp", "max"),
                evidence_count=("value", "size"),
                confidence=("confidence", "max"),
            )
            .reset_index()
            .sort_values(
                ["evidence_count", "confidence", "latest_timestamp"],
                ascending=[False, False, False],
            )
        )
        top_value = grouped.iloc[0]["value"]
        row = (
            subset[subset["value"] == top_value]
            .sort_values(["timestamp", "confidence"], ascending=[False, False])
            .iloc[0]
        )
        _append_fact(
            rows,
            category="location",
            attribute=attribute,
            value=row["value"],
            timestamp=row["timestamp"],
            chat_name=row.get("chat_name", ""),
            evidence_text=row.get("evidence_text", ""),
            context=row.get("context", ""),
            status=row.get("status", "current"),
            confidence=min(1.0, float(row["confidence"]) + 0.05),
            source_kind="summary",
        )

    partner_names = raw_facts[
        (raw_facts["category"] == "relationship")
        & (raw_facts["attribute"] == "partner_name")
    ].sort_values(["timestamp", "confidence"], ascending=[False, False])
    partner_status = raw_facts[
        (raw_facts["category"] == "relationship")
        & (raw_facts["attribute"] == "partner_status")
    ].sort_values(["timestamp", "confidence"], ascending=[False, False])

    latest_partner_name_ts = (
        pd.to_datetime(partner_names.iloc[0]["timestamp"])
        if not partner_names.empty
        else pd.NaT
    )
    latest_partner_status = partner_status.iloc[0] if not partner_status.empty else None
    latest_partner_status_ts = (
        pd.to_datetime(latest_partner_status["timestamp"])
        if latest_partner_status is not None
        else pd.NaT
    )

    if not partner_names.empty and (
        latest_partner_status is None
        or pd.isna(latest_partner_status_ts)
        or pd.isna(latest_partner_name_ts)
        or latest_partner_name_ts >= latest_partner_status_ts
        or latest_partner_status["value"] == "together"
    ):
        top_partner = partner_names.iloc[0]
        _append_fact(
            rows,
            category="relationship",
            attribute="partner_current_name",
            value=top_partner["value"],
            timestamp=top_partner["timestamp"],
            chat_name=top_partner.get("chat_name", ""),
            evidence_text=top_partner.get("evidence_text", ""),
            context=top_partner.get("context", ""),
            status="current",
            confidence=min(1.0, float(top_partner["confidence"]) + 0.1),
            source_kind="summary",
        )
        _append_fact(
            rows,
            category="relationship",
            attribute="partner_status_current",
            value="together",
            timestamp=top_partner["timestamp"],
            status="current",
            confidence=0.82,
            source_kind="summary",
        )
    elif latest_partner_status is not None:
        _append_fact(
            rows,
            category="relationship",
            attribute="partner_status_current",
            value=str(latest_partner_status["value"]),
            timestamp=latest_partner_status["timestamp"],
            chat_name=latest_partner_status.get("chat_name", ""),
            evidence_text=latest_partner_status.get("evidence_text", ""),
            context=latest_partner_status.get("context", ""),
            status="current",
            confidence=min(
                1.0,
                float(latest_partner_status["confidence"]) + 0.08,
            ),
            source_kind="summary",
        )

    for attribute in ["friend_name", "best_friend_name", "sister_like_name"]:
        subset = raw_facts[
            (raw_facts["category"] == "relationship")
            & (raw_facts["attribute"] == attribute)
        ]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["value", "attribute"])
            .agg(
                evidence_count=("value", "size"),
                latest_timestamp=("timestamp", "max"),
                confidence=("confidence", "max"),
            )
            .reset_index()
            .sort_values(
                ["evidence_count", "latest_timestamp", "confidence"],
                ascending=[False, False, False],
            )
        )
        for row in grouped.head(6).itertuples():
            _append_fact(
                rows,
                category="relationship",
                attribute=attribute,
                value=row.value,
                timestamp=row.latest_timestamp,
                status="current",
                confidence=min(1.0, float(row.confidence) + 0.05),
                source_kind="summary",
                evidence_count=int(row.evidence_count),
            )

    work_subset = raw_facts[raw_facts["category"] == "work"]
    if not work_subset.empty:
        current_companies = work_subset[
            (work_subset["attribute"] == "company_name")
            & (work_subset["status"] == "current")
        ].sort_values(["timestamp", "confidence"], ascending=[False, False])
        if not current_companies.empty:
            row = current_companies.iloc[0]
            _append_fact(
                rows,
                category="work",
                attribute="current_company_name",
                value=row["value"],
                timestamp=row["timestamp"],
                chat_name=row.get("chat_name", ""),
                evidence_text=row.get("evidence_text", ""),
                context=row.get("context", ""),
                status="current",
                confidence=min(1.0, float(row["confidence"]) + 0.1),
                source_kind="summary",
            )

        unique_companies = (
            work_subset[work_subset["attribute"] == "company_name"]
            .groupby("value")
            .agg(
                latest_timestamp=("timestamp", "max"),
                evidence_count=("value", "size"),
                confidence=("confidence", "max"),
            )
            .reset_index()
            .sort_values(
                ["evidence_count", "latest_timestamp", "confidence"],
                ascending=[False, False, False],
            )
        )
        for row in unique_companies.head(10).itertuples():
            _append_fact(
                rows,
                category="work",
                attribute="company_name",
                value=row.value,
                timestamp=row.latest_timestamp,
                status="summary",
                confidence=min(1.0, float(row.confidence) + 0.05),
                source_kind="summary",
                evidence_count=int(row.evidence_count),
            )

        for attribute in ["job_role", "function", "schedule"]:
            grouped = (
                work_subset[work_subset["attribute"] == attribute]
                .groupby("value")
                .agg(
                    latest_timestamp=("timestamp", "max"),
                    evidence_count=("value", "size"),
                    confidence=("confidence", "max"),
                )
                .reset_index()
                .sort_values(
                    ["evidence_count", "latest_timestamp", "confidence"],
                    ascending=[False, False, False],
                )
            )
            for row in grouped.head(8).itertuples():
                _append_fact(
                    rows,
                    category="work",
                    attribute=attribute,
                    value=row.value,
                    timestamp=row.latest_timestamp,
                    status="summary",
                    confidence=min(1.0, float(row.confidence) + 0.04),
                    source_kind="summary",
                    evidence_count=int(row.evidence_count),
                )

    pet_subset = raw_facts[raw_facts["category"] == "pet"]
    if not pet_subset.empty:
        for attribute in ["pet_name", "pet_type"]:
            grouped = (
                pet_subset[pet_subset["attribute"] == attribute]
                .groupby("value")
                .agg(
                    latest_timestamp=("timestamp", "max"),
                    evidence_count=("value", "size"),
                    confidence=("confidence", "max"),
                )
                .reset_index()
                .sort_values(
                    ["evidence_count", "latest_timestamp", "confidence"],
                    ascending=[False, False, False],
                )
            )
            for row in grouped.head(6).itertuples():
                _append_fact(
                    rows,
                    category="pet",
                    attribute=attribute,
                    value=row.value,
                    timestamp=row.latest_timestamp,
                    status="summary",
                    confidence=min(1.0, float(row.confidence) + 0.05),
                    source_kind="summary",
                    evidence_count=int(row.evidence_count),
                )

    own_texts = (
        messages_df.loc[messages_df["is_me"].fillna(False), "text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    opener_counts: Counter[str] = Counter()
    laugh_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()

    for text in own_texts:
        normalized = _normalize_value(text).casefold()
        if not normalized or "http" in normalized:
            continue

        tokens = normalized.split()
        if tokens:
            opener = tokens[0].strip("¿?¡!.,;:")
            if opener in STYLE_OPENERS:
                opener_counts[opener] += 1

        if re.search(r"w?aja+j+", normalized):
            laugh_counts["jajaja"] += 1
        if "xd" in normalized:
            laugh_counts["xd"] += 1
        if "jeje" in normalized:
            laugh_counts["jeje"] += 1

        if len(tokens) <= 4:
            phrase_counts[normalized] += 1

    for value, count in opener_counts.most_common(settings.style_top_k):
        if count < settings.style_min_count:
            continue
        _append_fact(
            rows,
            category="style",
            attribute="opener",
            value=value,
            timestamp=timestamp_max,
            status="habitual",
            confidence=0.72,
            source_kind="summary",
            evidence_count=count,
        )

    for value, count in laugh_counts.most_common(4):
        if count < settings.style_min_count:
            continue
        _append_fact(
            rows,
            category="style",
            attribute="laugh_marker",
            value=value,
            timestamp=timestamp_max,
            status="habitual",
            confidence=0.72,
            source_kind="summary",
            evidence_count=count,
        )

    for value, count in phrase_counts.most_common(settings.style_top_k):
        if count < settings.style_min_count:
            continue
        _append_fact(
            rows,
            category="style",
            attribute="phrase",
            value=value,
            timestamp=timestamp_max,
            status="habitual",
            confidence=0.68,
            source_kind="summary",
            evidence_count=count,
        )

    summary_facts = pd.DataFrame(rows)
    if summary_facts.empty:
        return summary_facts
    summary_facts = summary_facts.sort_values(
        ["category", "attribute", "timestamp", "confidence"],
        ascending=[True, True, False, False],
    )
    summary_facts = summary_facts.drop_duplicates(
        subset=["category", "attribute", "value_key", "source_kind"],
        keep="first",
    )
    return summary_facts.reset_index(drop=True)


def _build_summary_dict(profile_facts: pd.DataFrame) -> dict[str, object]:
    summary_rows = profile_facts[
        profile_facts["source_kind"].eq("summary")
    ].copy()
    out: dict[str, object] = {}
    for category, group in summary_rows.groupby("category"):
        out[category] = {}
        for attribute, attr_group in group.groupby("attribute"):
            values = attr_group.sort_values(
                ["status", "timestamp", "confidence"],
                ascending=[True, False, False],
            )["value"].tolist()
            out[category][attribute] = values
    return out


def build_profile_artifacts(
    messages_df: pd.DataFrame,
    *,
    my_name: str,
    settings: ProfileBuildSettings | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    settings = settings or ProfileBuildSettings()
    raw_facts = _extract_raw_profile_facts(
        messages_df,
        my_name=my_name,
        settings=settings,
    )
    summary_facts = _make_summary_rows(
        raw_facts,
        messages_df=messages_df,
        my_name=my_name,
        settings=settings,
    )

    parts = []
    if not raw_facts.empty:
        parts.append(raw_facts)
    if not summary_facts.empty:
        parts.append(summary_facts)

    if parts:
        profile_facts = pd.concat(parts, ignore_index=True)
        profile_facts = profile_facts.sort_values(
            ["source_kind", "timestamp", "confidence"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    else:
        profile_facts = pd.DataFrame(
            columns=[
                "category",
                "attribute",
                "value",
                "value_key",
                "status",
                "confidence",
                "source_kind",
                "evidence_count",
                "timestamp",
                "chat_name",
                "evidence_text",
                "context",
                "source_id",
            ]
        )

    return profile_facts, _build_summary_dict(profile_facts)


def build_or_load_profile_artifacts(
    messages_df: pd.DataFrame,
    *,
    data_dir: Path,
    my_name: str,
    force: bool = False,
    settings: ProfileBuildSettings | None = None,
) -> dict[str, object]:
    settings = settings or ProfileBuildSettings()
    facts_path = data_dir / PROFILE_FACTS_FILE_NAME
    summary_path = data_dir / PROFILE_SUMMARY_FILE_NAME
    manifest_path = data_dir / PROFILE_MANIFEST_NAME
    current_manifest = {
        "version": settings.build_version,
        "rows": int(len(messages_df)),
        "my_name": my_name,
    }

    cache_hit = (
        not force
        and facts_path.exists()
        and summary_path.exists()
        and load_json(manifest_path, {}) == current_manifest
    )
    if cache_hit:
        return {
            "profile_facts": pd.read_parquet(facts_path),
            "profile_summary": load_json(summary_path, {}),
            "cache_hit": True,
        }

    profile_facts, profile_summary = build_profile_artifacts(
        messages_df,
        my_name=my_name,
        settings=settings,
    )
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    profile_facts.to_parquet(facts_path, index=False)
    save_json(summary_path, profile_summary)
    save_json(manifest_path, current_manifest)
    return {
        "profile_facts": profile_facts,
        "profile_summary": profile_summary,
        "cache_hit": False,
    }
