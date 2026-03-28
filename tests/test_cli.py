import argparse

import pytest

from wharagbot.cli import _run_ingest


def test_run_ingest_requires_existing_chat_directory(tmp_path):
    args = argparse.Namespace(
        project_root=tmp_path,
        chats_dir=tmp_path / "no-existe",
        my_name="Wenceslao",
    )

    with pytest.raises(SystemExit, match="No existe el directorio de chats"):
        _run_ingest(args)
