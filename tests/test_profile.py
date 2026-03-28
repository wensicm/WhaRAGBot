import pandas as pd

from wharagbot.profile import build_profile_artifacts, retrieve_profile_facts


def _cleaned_messages_fixture() -> pd.DataFrame:
    rows = [
        ("2025-01-01 10:00:00", "Wenceslao", "Mi comida favorita es el VIPS"),
        ("2025-01-02 10:00:00", "Wenceslao", "Mi color favorito es azul"),
        ("2025-01-03 10:00:00", "Wenceslao", "Mi churri es Adri"),
        ("2025-01-04 10:00:00", "Wenceslao", "Mis amigos David y Alberto"),
        ("2025-01-05 10:00:00", "Wenceslao", "Mi mejor amiga es Eli"),
        ("2025-01-06 10:00:00", "Wenceslao", "Trabajo en Borneo"),
        ("2025-01-07 10:00:00", "Wenceslao", "Trabajé en VIPS"),
        ("2025-01-08 10:00:00", "Wenceslao", "Trabajo de camarero"),
        ("2025-01-09 10:00:00", "Wenceslao", "Mi gato Kaleta"),
        ("2025-01-10 10:00:00", "Wenceslao", "Holi"),
        ("2025-01-11 10:00:00", "Wenceslao", "Holi"),
        ("2025-01-12 10:00:00", "Wenceslao", "Holi"),
        ("2025-01-13 10:00:00", "Wenceslao", "jajaja"),
        ("2025-01-14 10:00:00", "Wenceslao", "jajaja"),
        ("2025-01-15 10:00:00", "Wenceslao", "jajaja"),
    ]
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(timestamp),
                "chat_name": "Ana",
                "sender": sender,
                "text": text,
                "is_me": True,
            }
            for timestamp, sender, text in rows
        ]
    )


def test_build_profile_artifacts_extracts_core_profile_data():
    profile_facts, profile_summary = build_profile_artifacts(
        _cleaned_messages_fixture(),
        my_name="Wenceslao",
    )

    assert not profile_facts.empty
    assert profile_summary["identity"]["first_name"] == ["Wenceslao"]
    assert profile_summary["preference"]["favorite_food"] == ["el VIPS"]
    assert profile_summary["preference"]["favorite_color"] == ["azul"]
    assert profile_summary["relationship"]["partner_current_name"] == ["Adri"]
    assert "Borneo" in profile_summary["work"]["company_name"]
    assert "VIPS" in profile_summary["work"]["company_name"]
    assert "camarero" in profile_summary["work"]["job_role"]
    assert profile_summary["pet"]["pet_name"] == ["Kaleta"]
    assert "holi" in profile_summary["style"]["opener"]


def test_retrieve_profile_facts_prefers_partner_current_name():
    profile_facts, _ = build_profile_artifacts(
        _cleaned_messages_fixture(),
        my_name="Wenceslao",
    )

    hits = retrieve_profile_facts(
        profile_facts,
        query="Como se llama tu pareja actual?",
        k=5,
    )

    assert hits
    assert hits[0]["attribute"] == "partner_current_name"
    assert hits[0]["value"] == "Adri"


def test_build_profile_artifacts_filters_ambiguous_relationships():
    messages = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-08-22 22:07:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": (
                    "Vas a conocer a mi churri y a David y Alberto mis amigos "
                    "controladores que te he hablado de ellos"
                ),
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-08-22 22:08:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Que siempre os menciono a Adri pero aún no os conoce jajajaja",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-08-22 22:09:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Mi mejor amiga tiene novio desde hace 8 meses",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-08-22 22:10:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Mi mejor amiga es Eli",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-08-22 22:11:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Pero recuerda que no eres mi padre!! Que eres mi churri",
                "is_me": True,
            },
        ]
    )

    profile_facts, profile_summary = build_profile_artifacts(
        messages,
        my_name="Wenceslao",
    )

    partner_names = set(
        profile_facts.loc[
            profile_facts["attribute"].eq("partner_name"),
            "value",
        ].tolist()
    )
    assert "Adri" in partner_names
    assert "Alberto" not in partner_names
    assert "Que" not in partner_names
    assert profile_summary["relationship"]["partner_current_name"] == ["Adri"]
    assert profile_summary["relationship"]["best_friend_name"] == ["Eli"]


def test_build_profile_artifacts_filters_non_place_and_bad_job_roles():
    messages = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": (
                    "La verdad es que es una pena que desde que vivo en Las Palmas "
                    "nunca quedo con ella"
                ),
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-02 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Pues vivo en Miraflores de la Sierra",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-03 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Trabajo en la hostelería y vivo en una ansiedad constante",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-04 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Imaginate si lo vivo en persona",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-05 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Yo soy de block fácil",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-06 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Soy de ver las cosas de forma largoplacista",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-07 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Trabajo de camarero",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-08 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Otra opción sería buscar un trabajo de camarero los fines",
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-09 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": (
                    "Y casualmente buscando videos de como trabajo de cocinero "
                    "para CCOO me encontré esto"
                ),
                "is_me": True,
            },
            {
                "timestamp": pd.Timestamp("2025-01-10 10:00:00"),
                "chat_name": "Ana",
                "sender": "Wenceslao",
                "text": "Trabajo de 21 a 25",
                "is_me": True,
            },
        ]
    )

    profile_facts, profile_summary = build_profile_artifacts(
        messages,
        my_name="Wenceslao",
    )

    lives_in_values = set(
        profile_facts.loc[
            profile_facts["attribute"].eq("lives_in"),
            "value",
        ].tolist()
    )
    assert "Las Palmas" in lives_in_values
    assert "Miraflores de la Sierra" in lives_in_values
    assert "una ansiedad constante" not in lives_in_values
    assert "persona" not in lives_in_values
    assert "from_place" not in profile_summary.get("location", {})

    job_roles = set(
        profile_facts.loc[
            profile_facts["attribute"].eq("job_role"),
            "value",
        ].tolist()
    )
    assert "camarero" in job_roles
    assert "camarero los fines" not in job_roles
    assert "cocinero" not in job_roles
    assert "21 a 25" not in job_roles
    assert profile_summary["work"]["job_role"] == ["camarero", "hostelería"]
