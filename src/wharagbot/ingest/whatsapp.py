from __future__ import annotations

from pathlib import Path
import re
import zipfile

import pandas as pd


INPUT_COLUMNS = ["timestamp", "sender", "text", "chat_name"]

START_PATTERNS = [
    re.compile(
        r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), "
        r"(?P<time>\d{1,2}:\d{2}) - "
        r"(?P<sender>[^:]+): (?P<text>.*)$"
    ),
    re.compile(
        r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), "
        r"(?P<time>\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?))\] "
        r"(?P<sender>[^:]+): (?P<text>.*)$"
    ),
]

SYSTEM_START_PATTERNS = [
    re.compile(
        r"^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - .*$"
    ),
    re.compile(
        r"^\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)\] .*$"
    ),
]

SYSTEM_SNIPPETS = tuple(
    snippet.casefold()
    for snippet in (
        "cifrado de extremo a extremo",
        "cambió el asunto",
        "cambió la foto",
        "cambió la descripción",
        "creó este grupo",
        "te añadieron",
        "mensaje eliminado",
        "multimedia omitido",
    )
)


def empty_messages_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INPUT_COLUMNS)


def match_start(line: str):
    for pattern in START_PATTERNS:
        match = pattern.match(line)
        if match:
            return match
    return None


def is_system_start(line: str) -> bool:
    return any(pattern.match(line) for pattern in SYSTEM_START_PATTERNS)


def parse_chat_text(text: str, chat_name: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip("\ufeff")
        match = match_start(line)
        if match:
            if current:
                rows.append(current)
            current = {
                "date": match.group("date"),
                "time": match.group("time"),
                "sender": match.group("sender").strip(),
                "text": match.group("text").strip(),
                "chat_name": chat_name,
            }
            continue

        if is_system_start(line):
            if current:
                rows.append(current)
                current = None
            continue

        if current:
            current["text"] += " " + line.strip()

    if current:
        rows.append(current)

    if not rows:
        return empty_messages_frame()

    dataframe = pd.DataFrame(rows)
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["date"] + " " + dataframe["time"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    dataframe = dataframe.dropna(subset=["timestamp"]).copy()
    if dataframe.empty:
        return empty_messages_frame()

    mask = dataframe["text"].astype(str).map(
        lambda value: not any(
            snippet in value.casefold() for snippet in SYSTEM_SNIPPETS
        )
    )
    dataframe = dataframe[mask].copy()
    dataframe["text"] = dataframe["text"].astype(str).str.replace(
        "\u202f",
        " ",
        regex=False,
    )
    return dataframe[INPUT_COLUMNS].reset_index(drop=True)


def parse_zip_chat(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zip_file:
        txt_files = [
            name for name in zip_file.namelist() if name.lower().endswith(".txt")
        ]
        if not txt_files:
            return empty_messages_frame()

        raw = zip_file.read(txt_files[0])
        decoded_text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded_text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise UnicodeDecodeError(
                "whatsapp",
                raw,
                0,
                len(raw),
                f"No se pudo decodificar {zip_path.name}",
            )

    return parse_chat_text(decoded_text, chat_name=zip_path.stem)


def parse_csv_chat(csv_path: Path, my_name: str) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(csv_path)
    except Exception:
        return empty_messages_frame()

    required = {"question", "wenceslao_answer"}
    if not required.issubset(set(dataframe.columns)):
        return empty_messages_frame()

    contact_col = "contact" if "contact" in dataframe.columns else None
    question_dt_col = "question_dt" if "question_dt" in dataframe.columns else None
    answer_dt_col = "answer_dt" if "answer_dt" in dataframe.columns else None

    rows = []
    for _, row in dataframe.iterrows():
        contact = str(row.get(contact_col, "Contacto")) if contact_col else "Contacto"
        chat_name = (
            f"Chat CSV con {contact}" if contact else f"Chat CSV {csv_path.stem}"
        )

        question_text = str(row.get("question", "") or "").strip()
        answer_text = str(row.get("wenceslao_answer", "") or "").strip()
        question_ts = pd.to_datetime(
            row.get(question_dt_col),
            dayfirst=True,
            errors="coerce",
        ) if question_dt_col else pd.NaT
        answer_ts = pd.to_datetime(
            row.get(answer_dt_col),
            dayfirst=True,
            errors="coerce",
        ) if answer_dt_col else pd.NaT

        if question_text:
            rows.append(
                {
                    "timestamp": question_ts,
                    "sender": contact,
                    "text": question_text,
                    "chat_name": chat_name,
                }
            )
        if answer_text:
            rows.append(
                {
                    "timestamp": answer_ts,
                    "sender": my_name,
                    "text": answer_text,
                    "chat_name": chat_name,
                }
            )

    if not rows:
        return empty_messages_frame()

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["timestamp"]).copy()
    if out.empty:
        return empty_messages_frame()
    out["text"] = out["text"].astype(str).str.replace(
        "\u202f",
        " ",
        regex=False,
    )
    out = out.sort_values(["chat_name", "timestamp"]).reset_index(drop=True)
    return out[INPUT_COLUMNS]


def parse_input_file(path: Path, my_name: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return parse_zip_chat(path)
    if suffix == ".csv":
        return parse_csv_chat(path, my_name=my_name)
    return empty_messages_frame()


def list_input_files(chats_dir: Path) -> list[Path]:
    zip_files = sorted(chats_dir.glob("*.zip"))
    csv_files = sorted(chats_dir.glob("*.csv"))
    return zip_files + csv_files


def load_messages_from_directory(chats_dir: Path, my_name: str) -> pd.DataFrame:
    frames = []
    for path in list_input_files(chats_dir):
        frame = parse_input_file(path, my_name=my_name)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return empty_messages_frame()

    messages = pd.concat(frames, ignore_index=True)
    messages = messages.dropna(subset=["timestamp"])
    messages = messages.sort_values(["chat_name", "timestamp"])
    return messages.reset_index(drop=True)
