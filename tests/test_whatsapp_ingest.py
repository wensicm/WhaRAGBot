from pathlib import Path

from wharagbot.ingest.whatsapp import parse_chat_text, parse_csv_chat


def test_parse_chat_text_handles_multiline_and_system_messages():
    raw = (
        "28/03/24, 09:00 - Alice: Holi\n"
        "28/03/24, 09:01 - Wenceslao: Buenas\n"
        "seguimos luego\n"
        "28/03/24, 09:02 - Mensajes y llamadas están cifrados de extremo a "
        "extremo.\n"
        "28/03/24, 09:03 - Alice: Vale\n"
    )

    dataframe = parse_chat_text(raw, chat_name="Chat prueba")

    assert list(dataframe["sender"]) == ["Alice", "Wenceslao", "Alice"]
    assert dataframe.iloc[1]["text"] == "Buenas seguimos luego"
    assert dataframe["chat_name"].nunique() == 1


def test_parse_csv_chat_creates_interleaved_messages(tmp_path: Path):
    csv_path = tmp_path / "chat.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "contact,question,wenceslao_answer,question_dt,answer_dt"
                ),
                (
                    "Pepe,¿Vienes?,Si claro,28/03/2024 10:00,28/03/2024 10:01"
                ),
            ]
        )
    )

    dataframe = parse_csv_chat(csv_path, my_name="Wenceslao")

    assert list(dataframe["sender"]) == ["Pepe", "Wenceslao"]
    assert list(dataframe["text"]) == ["¿Vienes?", "Si claro"]
    assert dataframe.iloc[0]["chat_name"] == "Chat CSV con Pepe"
