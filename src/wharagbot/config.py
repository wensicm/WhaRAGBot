from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


DEFAULT_CTX_WINDOW = 4
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_GEN_MODEL = "gpt-4.1-mini"


def discover_project_root() -> Path:
    """Devuelve la raiz del repo a partir de este modulo."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    chats_dir: Path
    data_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    paths: ProjectPaths
    my_name: str
    ctx_window: int
    embed_model: str
    gen_model: str
    openai_api_key: str


def load_runtime_config(
    project_root: Path | None = None,
    env_file: Path | None = None,
    chats_dir: Path | None = None,
    my_name: str | None = None,
) -> RuntimeConfig:
    """Carga configuracion del proyecto desde `.env` y argumentos directos."""
    root = Path(project_root or discover_project_root()).resolve()
    dotenv_path = Path(env_file).resolve() if env_file else root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=True)

    resolved_chats_dir = chats_dir or Path(
        os.getenv("CHATS_ZIP_DIR", str(root / "Chats en .zip"))
    )
    if not resolved_chats_dir.is_absolute():
        resolved_chats_dir = (root / resolved_chats_dir).resolve()

    paths = ProjectPaths(
        root=root,
        chats_dir=resolved_chats_dir,
        data_dir=root / "data",
        index_dir=root / "index",
    )

    return RuntimeConfig(
        paths=paths,
        my_name=my_name or os.getenv("MY_NAME", "Tu Nombre"),
        ctx_window=int(os.getenv("CTX_WINDOW", DEFAULT_CTX_WINDOW)),
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        gen_model=os.getenv("GEN_MODEL", DEFAULT_GEN_MODEL),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )
