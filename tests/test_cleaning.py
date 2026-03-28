import pandas as pd

from wharagbot.cleaning.normalize import clean_messages


def test_clean_messages_resolves_identity_and_labels_rows():
    messages = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
                "sender": "Ana",
                "text": "¿Qué tal?",
                "chat_name": "Ana",
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
                "sender": "Wenceslao",
                "text": "Bien bien",
                "chat_name": "Ana",
            },
            {
                "timestamp": pd.Timestamp("2024-03-02 11:00:00"),
                "sender": "Wenceslao",
                "text": "Trabajo en remoto",
                "chat_name": "Trabajo",
            },
        ]
    )

    cleaned, resolution = clean_messages(messages, configured_name="Tu Nombre")

    assert resolution.name == "Wenceslao"
    assert resolution.source == "inferred_chat_coverage"
    assert int(cleaned["is_me"].sum()) == 2
    assert bool(cleaned.iloc[0]["is_question"]) is True
    assert bool(cleaned.iloc[2]["self_fact"]) is True


def test_clean_messages_marks_low_signal_and_deduplicates():
    messages = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
                "sender": "Wenceslao",
                "text": "ok",
                "chat_name": "Ana",
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
                "sender": "Wenceslao",
                "text": "ok",
                "chat_name": "Ana",
            },
        ]
    )

    cleaned, _ = clean_messages(messages, configured_name="Wenceslao")

    assert len(cleaned) == 1
    assert bool(cleaned.iloc[0]["is_low_signal"]) is True


def test_clean_messages_marks_personal_preferences_as_self_fact():
    messages = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-03-01 10:00:00"),
                "sender": "Wenceslao",
                "text": "Mi comida favorita es el vips",
                "chat_name": "Ana",
            },
            {
                "timestamp": pd.Timestamp("2024-03-01 10:01:00"),
                "sender": "Wenceslao",
                "text": "Mi mejor amiga es como una hermana para mí",
                "chat_name": "Ana",
            },
        ]
    )

    cleaned, _ = clean_messages(messages, configured_name="Wenceslao")

    assert cleaned["self_fact"].tolist() == [True, True]
