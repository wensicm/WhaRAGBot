from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import pandas as pd

from wharagbot.ingest.whatsapp import INPUT_COLUMNS, empty_messages_frame


ACK_WORDS = {
    "ok",
    "oki",
    "okey",
    "vale",
    "si",
    "sii",
    "sip",
    "aja",
    "mmm",
    "mm",
    "mhm",
    "jaja",
    "jajaja",
    "jeje",
    "jaj",
    "xd",
    "xD",
}
QUESTION_PREFIXES = (
    "que ",
    "como ",
    "por que ",
    "por qué ",
    "cuando ",
    "cuanto ",
    "donde ",
    "quien ",
    "cual ",
    "cuales ",
)
SELF_FACT_HINTS = [
    "me llamo",
    "mi nombre",
    "me llaman",
    "soy de",
    "vivo en",
    "trabajo",
    "trabajo en",
    "trabajé en",
    "trabaje en",
    "he trabajado",
    "curro",
    "curro en",
    "me dedico",
    "estudio",
    "tengo ",
    "mi comida favorita",
    "mi color favorito",
    "mi color preferido",
    "mi mejor amigo",
    "mi mejor amiga",
    "mi churri",
    "mi pareja",
    "mis ex",
    "mis exparejas",
    "me gusta",
    "me apasiona",
    "prefiero",
]
SELF_FACT_PATTERNS = [
    re.compile(
        r"\bmi\s+(?:comida|color|pelicula|serie|cancion|libro)\s+favorit[oa]s?\b",
        flags=re.I,
    ),
    re.compile(r"\bmi\s+mejor\s+amig[oa]\b", flags=re.I),
    re.compile(r"\bmi\s+(?:churri|pareja|novi[oa])\b", flags=re.I),
    re.compile(r"\bmis\s+ex(?:parejas)?\b", flags=re.I),
    re.compile(r"\bme\s+dedic[oa]\b", flags=re.I),
    re.compile(r"\b(?:he\s+trabajado|trabaj[ée]\s+en|curro\s+en)\b", flags=re.I),
    re.compile(r"\bme\s+gusta\b", flags=re.I),
    re.compile(r"\bme\s+apasiona\b", flags=re.I),
    re.compile(r"\bprefiero\b", flags=re.I),
]
MY_NAME_PLACEHOLDERS = {"", "tu nombre", "your name", "mi nombre"}

CLEAN_MESSAGE_COLUMNS = [
    *INPUT_COLUMNS,
    "sender_key",
    "is_me",
    "text_raw",
    "has_url",
    "is_question",
    "is_low_signal",
    "signal_score",
    "token_len",
    "self_fact",
]


@dataclass(frozen=True)
class IdentityResolution:
    name: str
    source: str


def normalize_text(text: str) -> str:
    normalized = str(text or "")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_sender_name(name: str) -> str:
    return unicodedata.normalize("NFKC", str(name or "")).strip()


def sender_key(name: str) -> str:
    return normalize_sender_name(name).casefold()


def has_url(text: str) -> bool:
    return bool(re.search(r"(https?://|www\.)", text or "", flags=re.I))


def is_question(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if "?" in normalized:
        return True
    return any(normalized.startswith(prefix) for prefix in QUESTION_PREFIXES)


def is_low_signal(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    if normalized in ACK_WORDS:
        return True
    if len(normalized) <= 2:
        return True
    if re.fullmatch(r"[\W_]+", normalized):
        return True
    if sum(character.isalnum() for character in normalized) == 0:
        return True
    return False


def signal_score(
    text: str,
    *,
    text_is_question: bool = False,
    text_has_url: bool = False,
) -> float:
    normalized = (text or "").strip()
    tokens = len(normalized.split())
    score = min(1.0, tokens / 12)
    if len(normalized) >= 80:
        score += 0.10
    if text_is_question:
        score += 0.05
    if text_has_url:
        score -= 0.10
    return max(0.0, min(1.0, score))


def is_self_fact(text: str) -> bool:
    normalized = (text or "").lower()
    return any(hint in normalized for hint in SELF_FACT_HINTS) or any(
        pattern.search(text or "") for pattern in SELF_FACT_PATTERNS
    )


def resolve_my_name(
    messages: pd.DataFrame,
    configured_name: str,
) -> IdentityResolution:
    configured_raw = normalize_sender_name(configured_name)
    configured_key = sender_key(configured_raw)

    if messages.empty or "sender" not in messages.columns:
        fallback = configured_raw or "Tu Nombre"
        return IdentityResolution(name=fallback, source="fallback_default")

    senders = messages["sender"].astype(str).map(normalize_sender_name)
    sender_keys = senders.map(sender_key)

    if (
        configured_key
        and configured_key not in MY_NAME_PLACEHOLDERS
        and (sender_keys == configured_key).any()
    ):
        chosen = senders[sender_keys == configured_key].value_counts().idxmax()
        return IdentityResolution(name=str(chosen), source="config_match")

    tmp = pd.DataFrame(
        {
            "sender": senders,
            "sender_key": sender_keys,
            "chat_name": messages["chat_name"].astype(str),
        }
    )
    tmp = tmp[tmp["sender_key"] != ""]
    if tmp.empty:
        fallback = configured_raw or "Tu Nombre"
        return IdentityResolution(name=fallback, source="fallback_default")

    coverage = (
        tmp.groupby("sender_key")["chat_name"]
        .nunique()
        .sort_values(ascending=False)
    )
    top_key = str(coverage.index[0])
    top_coverage = int(coverage.iloc[0])
    total_chats = max(1, int(tmp["chat_name"].nunique()))

    if total_chats >= 2 and top_coverage >= max(2, int(total_chats * 0.5)):
        chosen = (
            tmp.loc[tmp["sender_key"] == top_key, "sender"]
            .value_counts()
            .idxmax()
        )
        return IdentityResolution(
            name=str(chosen),
            source="inferred_chat_coverage",
        )

    if configured_raw and configured_key not in MY_NAME_PLACEHOLDERS:
        return IdentityResolution(name=configured_raw, source="config_no_match")

    chosen = tmp["sender"].value_counts().idxmax()
    return IdentityResolution(name=str(chosen), source="inferred_top_count")


def clean_messages(
    messages: pd.DataFrame,
    configured_name: str,
) -> tuple[pd.DataFrame, IdentityResolution]:
    if messages.empty:
        resolution = resolve_my_name(messages, configured_name)
        return empty_messages_frame().reindex(columns=CLEAN_MESSAGE_COLUMNS), resolution

    cleaned = messages.copy()
    cleaned["sender"] = cleaned["sender"].astype(str).map(normalize_sender_name)
    cleaned["sender_key"] = cleaned["sender"].map(sender_key)

    resolution = resolve_my_name(cleaned, configured_name)
    my_name_key = sender_key(resolution.name)
    cleaned["is_me"] = cleaned["sender_key"] == my_name_key if my_name_key else False

    cleaned["text_raw"] = cleaned["text"].astype(str)
    cleaned["text"] = cleaned["text_raw"].map(normalize_text)
    cleaned = cleaned[cleaned["text"].str.len() > 0].copy()

    cleaned["has_url"] = cleaned["text"].map(has_url)
    cleaned["is_question"] = cleaned["text"].map(is_question)
    cleaned["is_low_signal"] = cleaned["text"].map(is_low_signal)
    cleaned["signal_score"] = cleaned.apply(
        lambda row: signal_score(
            row["text"],
            text_is_question=bool(row["is_question"]),
            text_has_url=bool(row["has_url"]),
        ),
        axis=1,
    )
    cleaned["token_len"] = cleaned["text"].str.split().str.len()
    cleaned["self_fact"] = cleaned["is_me"] & cleaned["text"].map(is_self_fact)

    cleaned = cleaned.drop_duplicates(
        subset=["chat_name", "timestamp", "sender", "text"]
    )
    cleaned = cleaned.sort_values(["chat_name", "timestamp"]).reset_index(drop=True)
    return cleaned.reindex(columns=CLEAN_MESSAGE_COLUMNS), resolution
