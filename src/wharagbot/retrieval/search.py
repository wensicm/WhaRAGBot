from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import faiss
import numpy as np
import pandas as pd

from wharagbot.embeddings.encoder import build_embedder, encode_queries


REQUIRED_INDEX_ARTIFACTS = (
    "response_units.parquet",
    "style_units.parquet",
    "fact_units.parquet",
    "response.index",
    "style.index",
    "fact.index",
    "response_embeddings.npy",
    "style_embeddings.npy",
    "fact_embeddings.npy",
)

FACT_QUERY_STOPWORDS = {
    "a",
    "al",
    "algo",
    "como",
    "con",
    "cual",
    "cuales",
    "cuanto",
    "cuantos",
    "de",
    "del",
    "donde",
    "el",
    "en",
    "es",
    "has",
    "hoy",
    "la",
    "las",
    "los",
    "me",
    "mi",
    "mis",
    "por",
    "que",
    "quien",
    "se",
    "su",
    "te",
    "tu",
    "tus",
    "un",
    "una",
    "y",
}

COMPANY_NAME_STOPWORDS = {
    "aqui",
    "camarero",
    "casa",
    "centro",
    "cosa",
    "cosas",
    "datos",
    "equipo",
    "empresa",
    "empleo",
    "gente",
    "hosteleria",
    "abril",
    "agosto",
    "diciembre",
    "paro",
    "enero",
    "persona",
    "febrero",
    "julio",
    "junio",
    "marzo",
    "mayo",
    "noviembre",
    "octubre",
    "programacion",
    "restaurante",
    "septiembre",
    "sitio",
    "sitios",
    "trabajo",
}

KNOWN_COMPANY_ALIASES = {
    "borneo": "Borneo",
    "mar": "Mar",
    "mar gastrotasca": "Mar Gastrotasca",
    "vips": "VIPS",
}


@dataclass
class DualIndexBundle:
    response_units: pd.DataFrame
    style_units: pd.DataFrame
    response_index: faiss.Index
    style_index: faiss.Index
    response_embeddings: np.ndarray
    style_embeddings: np.ndarray
    fact_units: pd.DataFrame | None = None
    fact_index: faiss.Index | None = None
    fact_embeddings: np.ndarray | None = None
    embedder: object | None = None

    def __post_init__(self) -> None:
        if self.fact_units is None:
            self.fact_units = pd.DataFrame()
        self.max_ts_response = pd.to_datetime(
            self.response_units["timestamp"],
            errors="coerce",
        ).max()
        self.max_ts_style = pd.to_datetime(
            self.style_units["timestamp"],
            errors="coerce",
        ).max()
        if self.fact_units.empty:
            self.max_ts_fact = pd.NaT
        else:
            self.max_ts_fact = pd.to_datetime(
                self.fact_units["timestamp"],
                errors="coerce",
            ).max()

        style_anchor_order = self.style_units.copy()
        if "style_source" not in style_anchor_order.columns:
            style_anchor_order["style_source"] = "memory"
        style_anchor_order["timestamp_dt"] = pd.to_datetime(
            style_anchor_order["timestamp"],
            errors="coerce",
        )
        style_anchor_order["style_source_rank"] = (
            style_anchor_order["style_source"].eq("micro").astype(int)
        )
        self.style_units_anchor_order = style_anchor_order.sort_values(
            ["style_source_rank", "timestamp_dt", "signal_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)


def load_dual_index_bundle(
    *,
    index_dir: Path,
    embed_model: str,
    embedder=None,
) -> DualIndexBundle:
    missing_files = [
        name
        for name in REQUIRED_INDEX_ARTIFACTS
        if not (index_dir / name).exists()
    ]
    if missing_files:
        missing_display = ", ".join(missing_files)
        raise FileNotFoundError(
            "Falta el bundle RAG en "
            f"{index_dir}. Ejecuta `wharagbot build-rag` primero. "
            f"Archivos ausentes: {missing_display}"
        )

    if embedder is None:
        embedder = build_embedder(embed_model)

    return DualIndexBundle(
        response_units=pd.read_parquet(index_dir / "response_units.parquet"),
        style_units=pd.read_parquet(index_dir / "style_units.parquet"),
        fact_units=pd.read_parquet(index_dir / "fact_units.parquet"),
        response_index=faiss.read_index(str(index_dir / "response.index")),
        style_index=faiss.read_index(str(index_dir / "style.index")),
        fact_index=faiss.read_index(str(index_dir / "fact.index")),
        response_embeddings=np.load(index_dir / "response_embeddings.npy"),
        style_embeddings=np.load(index_dir / "style_embeddings.npy"),
        fact_embeddings=np.load(index_dir / "fact_embeddings.npy"),
        embedder=embedder,
    )


def norm_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"""[¿?¡!.,;:\-_"'()\[\]]""", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def contact_hint(bundle: DualIndexBundle, query: str) -> str | None:
    query_low = query.lower()
    chat_series = [bundle.response_units["chat_name"], bundle.style_units["chat_name"]]
    if bundle.fact_units is not None and not bundle.fact_units.empty:
        chat_series.append(bundle.fact_units["chat_name"])
    all_chats = pd.concat(chat_series, ignore_index=True).dropna().unique().tolist()
    for chat_name in all_chats:
        if str(chat_name).lower() in query_low:
            return str(chat_name)
    return None


def recency_boost(ts, max_ts, *, alpha: float = 0.05) -> float:
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts) or pd.isna(max_ts):
        return 0.0
    days = max((max_ts - ts).days, 0)
    return max(0.0, alpha - days * 0.001)


def response_query_text(query: str) -> str:
    return f"Mensaje nuevo del contacto:\n{str(query or '').strip()}"


def style_query_text(query: str, contact: str | None = None) -> str:
    extra = f"\n\nChat objetivo:\n{contact}" if contact else ""
    return (
        "Como suelo responder por WhatsApp a este mensaje. "
        "Busca tono, longitud, muletillas y ritmo.\n\n"
        f"Mensaje del contacto:\n{str(query or '').strip()}{extra}"
    )


def fact_query_text(query: str) -> str:
    return (
        "Hechos y datos sobre mi que ayuden a responder esta pregunta.\n\n"
        f"Pregunta:\n{str(query or '').strip()}"
    )


def query_is_personal_fact(query: str) -> bool:
    normalized = norm_text(query)
    hints = (
        "a que te dedicas",
        "donde has trabajado",
        "donde trabajas",
        "cual es tu",
        "como se llama tu",
        "quien es tu",
        "cuantos ex",
        "cuantas ex",
        "tu comida favorita",
        "tu color favorito",
        "tu mejor amigo",
        "tu mejor amiga",
        "tu pareja",
        "tu churri",
        "te gusta",
        "te apasiona",
        "de donde eres",
        "donde vives",
    )
    return any(hint in normalized for hint in hints)


def query_requests_company_names(query: str) -> bool:
    normalized = norm_text(query)
    hints = (
        "nombre de la empresa",
        "nombre de empresa",
        "sitios donde has trabajado",
        "donde has trabajado",
        "empresas donde has trabajado",
        "empresa",
    )
    return any(hint in normalized for hint in hints)


def normalize_company_name(name: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(name or "").strip(" .,;:()[]{}\"'"))
    cleaned = re.split(
        r"(?i)\s+(?:y|pero|aunque|porque|donde|cuando|mientras)\s+",
        cleaned,
        maxsplit=1,
    )[0].strip()
    if not cleaned:
        return None

    normalized = norm_text(cleaned)
    if not normalized or normalized in COMPANY_NAME_STOPWORDS:
        return None
    if len(normalized) < 3 and normalized != "mar":
        return None

    if normalized in KNOWN_COMPANY_ALIASES:
        return KNOWN_COMPANY_ALIASES[normalized]

    tokens = cleaned.split()
    if len(tokens) > 4:
        return None

    out_tokens = []
    for token in tokens:
        if token.isupper():
            out_tokens.append(token)
        else:
            out_tokens.append(token[:1].upper() + token[1:])
    return " ".join(out_tokens)


def extract_company_mentions(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    normalized = norm_text(raw)
    mentions: list[str] = []
    seen: set[str] = set()

    for needle, canonical in KNOWN_COMPANY_ALIASES.items():
        if (
            re.search(rf"\b{re.escape(needle)}\b", normalized)
            and canonical not in seen
        ):
            mentions.append(canonical)
            seen.add(canonical)

    patterns = (
        r"\b(?i:trabajo|trabaj[ée]|trabaje|trabajaba|trabajando|"
        r"he trabajado|curro|curraba|contrataron(?:me)?|contratado|"
        r"alta en)\s+en\s+"
        r"([A-ZÁÉÍÓÚÑ0-9][\wÁÉÍÓÚÑ&.\-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ0-9][\wÁÉÍÓÚÑ&.\-]+){0,3})",
        r"\b(?i:entrevista)\s+(?:de|en)\s+"
        r"([A-ZÁÉÍÓÚÑ0-9][\wÁÉÍÓÚÑ&.\-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ0-9][\wÁÉÍÓÚÑ&.\-]+){0,3})",
        r"\b(?i:mi antiguo trabajo)\b.*?\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ&.\-]+"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ&.\-]+){0,3})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            candidate = normalize_company_name(match.group(1))
            if candidate and candidate not in seen:
                mentions.append(candidate)
                seen.add(candidate)

    return mentions


def fact_query_terms(query: str) -> list[str]:
    normalized = norm_text(query)
    terms: list[str] = []
    for token in normalized.split():
        if len(token) < 3 or token in FACT_QUERY_STOPWORDS:
            continue
        terms.append(token)
        if len(token) >= 6:
            terms.append(token[:6])

    extra: list[str] = []
    if any(stem in normalized for stem in ("dedic", "trabaj", "curro", "empresa")):
        extra.extend(
            [
                "trabaj",
                "curro",
                "empresa",
                "program",
                "codigo",
                "datos",
                "ingenier",
                "camarer",
                "vips",
                "borneo",
            ]
        )
    if any(stem in normalized for stem in ("comida", "favorit", "cen")):
        extra.extend(
            ["comida", "favorit", "pizza", "sushi", "hamburg", "vips"]
        )
    if "color" in normalized or "favorit" in normalized:
        extra.extend(["color", "favorit", "azul", "gris", "verde", "rojo"])
    if any(stem in normalized for stem in ("pareja", "churri", "novi")):
        extra.extend(["churri", "pareja", "novio", "novia", "adri"])
    if "ex" in normalized:
        extra.extend(["ex", "exparej"])
    if "mejor amig" in normalized or "hermana" in normalized:
        extra.extend(["mejor", "amig", "hermana"])

    seen = set()
    unique_terms = []
    for term in terms + extra:
        if term and term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms


def fact_keyword_score(text: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    normalized = norm_text(text)
    matched = 0
    score = 0.0
    for term in query_terms:
        if term and term in normalized:
            matched += 1
            score += 0.02 if len(term) < 6 else 0.03

    if matched >= 2:
        score += 0.04
    if matched >= 4:
        score += 0.04
    return min(score, 0.22)


def work_context_score(text: str) -> float:
    normalized = norm_text(text)
    stems = (
        "trabaj",
        "curro",
        "empleo",
        "empresa",
        "entrevista",
        "contrat",
        "paro",
        "despido",
        "hosteler",
        "turno",
        "alta",
        "camarer",
        "restaurante",
    )
    matched = sum(1 for stem in stems if stem in normalized)
    if matched == 0:
        return 0.0
    return min(0.06 * matched, 0.24)


def worked_company_evidence_score(text: str) -> float:
    normalized = norm_text(text)
    negative_hints = (
        "busco trabajo en",
        "buscar trabajo en",
        "entrevista en",
        "entrevista de",
        "me ofrecen",
        "oferta de trabajo",
    )
    if any(hint in normalized for hint in negative_hints):
        return 0.0

    positive_hints = (
        "trabajo en",
        "trabaje en",
        "trabajaba en",
        "he trabajado en",
        "trabajando en",
        "alta en",
        "mi antiguo trabajo",
        "hace un ano que trabajo en",
    )
    if any(hint in normalized for hint in positive_hints):
        return 0.22
    return 0.0


def employment_company_bonus(text: str) -> float:
    mentions = extract_company_mentions(text)
    if not mentions:
        return 0.0

    n_words = len(str(text or "").split())
    bonus = 0.18 + 0.05 * min(len(mentions), 3)
    if n_words <= 14:
        bonus += 0.08
    elif n_words <= 28:
        bonus += 0.04
    elif n_words >= 90:
        bonus -= 0.03
    return max(0.0, min(bonus, 0.38))


def lexical_fact_hits(
    bundle: DualIndexBundle,
    *,
    query: str,
    query_terms: list[str],
    k: int,
) -> list[dict[str, object]]:
    if bundle.fact_units is None or bundle.fact_units.empty:
        return []

    wants_company_names = query_requests_company_names(query)
    ranked = []
    for row in bundle.fact_units.to_dict("records"):
        fact_text = str(
            row.get("fact_text", row.get("response", "")) or ""
        ).strip()
        support_text = fact_text + "\n" + str(row.get("context", "") or "")
        keyword_score = fact_keyword_score(support_text, query_terms)
        company_mentions = extract_company_mentions(fact_text)
        work_score = (
            work_context_score(support_text) if wants_company_names else 0.0
        )
        worked_score = (
            worked_company_evidence_score(fact_text)
            if wants_company_names
            else 0.0
        )
        company_bonus = (
            employment_company_bonus(fact_text) if wants_company_names else 0.0
        )
        if wants_company_names and (
            not company_mentions
            or work_score <= 0
            or worked_score <= 0
        ):
            continue
        if keyword_score <= 0 and company_bonus <= 0:
            continue

        signal = float(row.get("signal_score", 0.4) or 0.4)
        row["score"] = 0.0
        row["keyword_score"] = keyword_score
        row["rank_score"] = (
            keyword_score
            + company_bonus
            + work_score
            + worked_score
            + (0.10 if bool(row.get("self_fact", False)) else 0.0)
            + 0.06 * max(0.25, signal)
            + recency_boost(
                row.get("timestamp"),
                bundle.max_ts_fact,
                alpha=0.05,
            )
        )
        if company_mentions:
            row["company_mentions"] = company_mentions
        row["retrieval_role"] = "fact_lexical"
        ranked.append(row)

    ranked = sorted(ranked, key=lambda item: item["rank_score"], reverse=True)
    dedup = []
    seen = set()
    for item in ranked:
        key = norm_text(str(item.get("fact_text", item.get("response", ""))))
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= k:
            break
    return dedup


def style_shape_bonus(n_words: int) -> float:
    if n_words <= 2:
        return 0.05
    if n_words <= 6:
        return 0.035
    if n_words <= 14:
        return 0.015
    return 0.0


def _default_encode_queries(bundle: DualIndexBundle, texts: list[str]) -> np.ndarray:
    if bundle.embedder is None:
        raise ValueError("No hay embedder para codificar queries")
    return encode_queries(bundle.embedder, texts)


def representative_style_hits(
    bundle: DualIndexBundle,
    *,
    k: int = 3,
    contact: str | None = None,
    seen_text: set[str] | None = None,
) -> list[dict[str, object]]:
    seen_text = set(seen_text or set())
    base = bundle.style_units_anchor_order
    if contact:
        scoped = base[base["chat_name"] == contact]
        if not scoped.empty:
            base = scoped

    out = []
    seen_sig = set()
    for row in base.to_dict("records"):
        text = str(row.get("style_text", "") or "").strip()
        text_key = norm_text(text)
        if not text_key or text_key in seen_text:
            continue

        signature = str(row.get("style_signature", "") or "")
        if signature in seen_sig and len(out) < max(1, k - 1):
            continue

        n_words = len(text.split())
        item = dict(row)
        item["score"] = 0.0
        item["rank_score"] = (
            0.18
            + recency_boost(item.get("timestamp"), bundle.max_ts_style, alpha=0.06)
            + style_shape_bonus(n_words)
            + (0.04 if item.get("style_source") == "micro" else 0.0)
        )
        item["retrieval_role"] = "style_anchor"
        out.append(item)
        seen_text.add(text_key)
        seen_sig.add(signature)
        if len(out) >= k:
            break
    return out


def retrieve_response_hits(
    bundle: DualIndexBundle,
    *,
    query: str,
    k: int = 6,
    min_score: float = 0.22,
    contact: str | None = None,
    encode_queries_fn=None,
) -> list[dict[str, object]]:
    encode_queries_fn = encode_queries_fn or (
        lambda texts: _default_encode_queries(bundle, texts)
    )
    q_vec = encode_queries_fn([response_query_text(query)])
    scores, indices = bundle.response_index.search(q_vec, max(60, k * 8))

    out = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        base_score = float(score)
        if base_score < min_score:
            continue

        row = bundle.response_units.iloc[index].to_dict()
        signal = float(row.get("signal_score", 0.4) or 0.4)
        rank_score = (
            base_score
            + 0.08 * signal
            + recency_boost(
                row.get("timestamp"),
                bundle.max_ts_response,
                alpha=0.05,
            )
            + (0.04 if contact and row.get("chat_name") == contact else 0.0)
        )

        gap = row.get("reply_gap_min")
        if gap is not None and not pd.isna(gap):
            gap = float(gap)
            rank_score += max(0.0, 0.03 - min(gap, 120.0) * 0.00025)

        row["score"] = base_score
        row["rank_score"] = rank_score
        row["retrieval_role"] = "response"
        out.append(row)

    out = sorted(out, key=lambda item: item["rank_score"], reverse=True)
    dedup = []
    seen = set()
    for item in out:
        key = item.get("source_id")
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= k:
            break
    return dedup


def retrieve_style_hits(
    bundle: DualIndexBundle,
    *,
    query: str,
    k: int = 10,
    min_score: float = 0.20,
    contact: str | None = None,
    encode_queries_fn=None,
) -> list[dict[str, object]]:
    encode_queries_fn = encode_queries_fn or (
        lambda texts: _default_encode_queries(bundle, texts)
    )
    q_vec = encode_queries_fn([style_query_text(query, contact)])
    scores, indices = bundle.style_index.search(q_vec, max(80, k * 8))

    ranked = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue
        base_score = float(score)
        if base_score < min_score:
            continue

        row = bundle.style_units.iloc[index].to_dict()
        signal = float(row.get("signal_score", 0.4) or 0.4)
        text = str(row.get("style_text", "") or "").strip()
        n_words = len(text.split())
        row["score"] = base_score
        row["rank_score"] = (
            base_score
            + 0.05 * max(0.25, signal)
            + recency_boost(
                row.get("timestamp"),
                bundle.max_ts_style,
                alpha=0.04,
            )
            + (0.03 if contact and row.get("chat_name") == contact else 0.0)
            + style_shape_bonus(n_words)
            + (0.04 if row.get("style_source") == "micro" else 0.0)
        )
        row["retrieval_role"] = "style"
        ranked.append(row)

    ranked = sorted(ranked, key=lambda item: item["rank_score"], reverse=True)
    buckets: dict[int, list[dict[str, object]]] = {}
    for item in ranked:
        cluster = int(item.get("style_cluster", -1))
        buckets.setdefault(cluster, []).append(item)

    diversified = []
    seen_text = set()
    clusters = list(buckets.keys())
    cursor = {cluster: 0 for cluster in clusters}
    while len(diversified) < k and clusters:
        next_clusters = []
        for cluster in clusters:
            index = cursor[cluster]
            arr = buckets[cluster]
            while index < len(arr):
                candidate = arr[index]
                index += 1
                text_key = norm_text(str(candidate.get("style_text", "")))
                if not text_key or text_key in seen_text:
                    continue
                seen_text.add(text_key)
                diversified.append(candidate)
                break
            cursor[cluster] = index
            if index < len(arr):
                next_clusters.append(cluster)
            if len(diversified) >= k:
                break
        clusters = next_clusters

    anchor_target = max(2, k // 3)
    semantic_target = max(1, k - anchor_target)
    selected = diversified[:semantic_target]
    selected_seen = {
        norm_text(str(item.get("style_text", "")))
        for item in selected
        if str(item.get("style_text", "")).strip()
    }
    anchors = representative_style_hits(
        bundle,
        k=anchor_target,
        contact=contact,
        seen_text=selected_seen,
    )
    selected.extend(anchors)
    selected_seen = {
        norm_text(str(item.get("style_text", "")))
        for item in selected
        if str(item.get("style_text", "")).strip()
    }

    for item in diversified[semantic_target:]:
        text_key = norm_text(str(item.get("style_text", "")))
        if not text_key or text_key in selected_seen:
            continue
        selected.append(item)
        selected_seen.add(text_key)
        if len(selected) >= k:
            break

    return selected[:k]


def retrieve_fact_hits(
    bundle: DualIndexBundle,
    *,
    query: str,
    k: int = 6,
    min_score: float = 0.18,
    encode_queries_fn=None,
) -> list[dict[str, object]]:
    if (
        bundle.fact_units is None
        or bundle.fact_units.empty
        or bundle.fact_index is None
    ):
        return []

    encode_queries_fn = encode_queries_fn or (
        lambda texts: _default_encode_queries(bundle, texts)
    )
    q_vec = encode_queries_fn([fact_query_text(query)])
    scores, indices = bundle.fact_index.search(q_vec, max(120, k * 12))
    query_terms = fact_query_terms(query)
    is_personal_fact = query_is_personal_fact(query)
    wants_company_names = query_requests_company_names(query)

    ranked = []
    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue

        base_score = float(score)
        threshold = min_score * (0.8 if is_personal_fact else 1.0)
        if base_score < threshold:
            continue

        row = bundle.fact_units.iloc[index].to_dict()
        fact_text = str(
            row.get("fact_text", row.get("response", "")) or ""
        ).strip()
        support_text = fact_text + "\n" + str(row.get("context", "") or "")
        keyword_score = fact_keyword_score(support_text, query_terms)
        company_mentions = extract_company_mentions(fact_text)
        work_score = (
            work_context_score(support_text) if wants_company_names else 0.0
        )
        worked_score = (
            worked_company_evidence_score(fact_text)
            if wants_company_names
            else 0.0
        )
        if is_personal_fact and keyword_score <= 0 and base_score < (min_score + 0.03):
            continue
        if wants_company_names and (
            not company_mentions
            or work_score <= 0
            or worked_score <= 0
        ):
            continue

        signal = float(row.get("signal_score", 0.4) or 0.4)
        rank_score = (
            base_score
            + 0.08 * max(0.25, signal)
            + (0.10 if bool(row.get("self_fact", False)) else 0.0)
            + keyword_score
            + work_score
            + worked_score
            + (
                employment_company_bonus(fact_text)
                if wants_company_names
                else 0.0
            )
            + recency_boost(
                row.get("timestamp"),
                bundle.max_ts_fact,
                alpha=0.05,
            )
            + (0.03 if is_personal_fact else 0.0)
        )
        if wants_company_names:
            if company_mentions:
                n_words = len(fact_text.split())
                rank_score += 0.08 if n_words <= 18 else 0.03
            elif len(fact_text.split()) > 60:
                rank_score -= 0.05
        row["score"] = base_score
        row["keyword_score"] = keyword_score
        row["rank_score"] = rank_score
        if company_mentions:
            row["company_mentions"] = company_mentions
        row["retrieval_role"] = "fact"
        ranked.append(row)

    ranked = sorted(ranked, key=lambda item: item["rank_score"], reverse=True)
    lexical = lexical_fact_hits(
        bundle,
        query=query,
        query_terms=query_terms,
        k=max(k, 8),
    )
    merged = ranked + lexical
    merged = sorted(merged, key=lambda item: item["rank_score"], reverse=True)
    dedup = []
    seen = set()
    for item in merged:
        key = norm_text(str(item.get("fact_text", item.get("response", ""))))
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= k:
            break
    return dedup


def retrieve_bundle(
    bundle: DualIndexBundle,
    *,
    query: str,
    k_response: int = 6,
    k_style: int = 8,
    k_fact: int = 6,
    min_score: float = 0.22,
    encode_queries_fn=None,
) -> dict[str, list[dict[str, object]]]:
    contact = contact_hint(bundle, query)
    response_hits = retrieve_response_hits(
        bundle,
        query=query,
        k=k_response,
        min_score=min_score,
        contact=contact,
        encode_queries_fn=encode_queries_fn,
    )
    style_hits = retrieve_style_hits(
        bundle,
        query=query,
        k=k_style,
        min_score=min_score * 0.9,
        contact=contact,
        encode_queries_fn=encode_queries_fn,
    )
    fact_hits = retrieve_fact_hits(
        bundle,
        query=query,
        k=k_fact,
        min_score=min_score * 0.82,
        encode_queries_fn=encode_queries_fn,
    )
    return {
        "response_hits": response_hits,
        "style_hits": style_hits,
        "fact_hits": fact_hits,
    }


def retrieve(
    bundle: DualIndexBundle,
    *,
    query: str,
    k_total: int = 10,
    min_score: float = 0.22,
    encode_queries_fn=None,
) -> list[dict[str, object]]:
    result = retrieve_bundle(
        bundle,
        query=query,
        k_response=max(4, k_total // 2),
        k_style=max(4, k_total),
        k_fact=max(4, k_total // 2),
        min_score=min_score,
        encode_queries_fn=encode_queries_fn,
    )
    merged = (
        result["fact_hits"]
        + result["response_hits"]
        + result["style_hits"]
    )
    merged = sorted(
        merged,
        key=lambda item: item.get("rank_score", item.get("score", 0.0)),
        reverse=True,
    )
    return merged[:k_total]
