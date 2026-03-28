from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from wharagbot.cleaning.normalize import clean_messages
from wharagbot.config import load_runtime_config
from wharagbot.ingest.whatsapp import (
    list_input_files,
    load_messages_from_directory,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_dataframe(dataframe: pd.DataFrame, target_path: Path) -> Path:
    _ensure_dir(target_path.parent)
    try:
        dataframe.to_parquet(target_path, index=False)
        return target_path
    except Exception:
        fallback_path = target_path.with_suffix(".csv")
        dataframe.to_csv(fallback_path, index=False)
        return fallback_path


def _load_messages_artifact(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    raise FileNotFoundError(
        f"No existe artefacto en {parquet_path.name} ni {csv_path.name}"
    )


def _ensure_existing_input_dir(chats_dir: Path) -> list[Path]:
    if not chats_dir.exists():
        raise SystemExit(
            f"No existe el directorio de chats: {chats_dir}\n"
            "Configura `CHATS_ZIP_DIR` en `.env` o usa "
            "`--chats-dir /ruta/a/tus/exportes`."
        )
    if not chats_dir.is_dir():
        raise SystemExit(
            f"La ruta de chats no es un directorio: {chats_dir}"
        )

    input_files = list_input_files(chats_dir)
    if not input_files:
        raise SystemExit(
            f"No hay exportes `.zip` o `.csv` en {chats_dir}\n"
            "Copia ahi tus exportes de WhatsApp o ajusta "
            "`CHATS_ZIP_DIR`/`--chats-dir`."
        )
    return input_files


def _load_required_artifact(base_path: Path, *, build_hint: str) -> pd.DataFrame:
    try:
        dataframe = _load_messages_artifact(base_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nEjecuta `{build_hint}` primero."
        ) from exc

    if dataframe.empty:
        raise SystemExit(
            f"El artefacto `{base_path.name}` esta vacio. "
            "Corrige la ingesta antes de continuar."
        )
    return dataframe


def _load_optional_artifact(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wharagbot",
        description="CLI inicial para ingesta y limpieza de exportes de WhatsApp.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Raiz del repo. Si se omite, se detecta automaticamente.",
    )
    parser.add_argument(
        "--chats-dir",
        type=Path,
        default=None,
        help="Directorio con exportes .zip/.csv.",
    )
    parser.add_argument(
        "--my-name",
        default=None,
        help="Nombre propio a usar como pista de identidad.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Parsea exportes y guarda messages.parquet.")
    subparsers.add_parser(
        "clean",
        help="Limpia messages.parquet/messages.csv y guarda messages_clean.parquet.",
    )
    subparsers.add_parser(
        "prepare-data",
        help="Ejecuta ingesta y limpieza de una vez.",
    )
    build_memory_parser = subparsers.add_parser(
        "build-memory",
        help="Construye memory_units, response_units y style_units.",
    )
    build_memory_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora manifests y fuerza reconstruccion.",
    )
    subparsers.add_parser(
        "build-index",
        help="Construye los indices duales FAISS a partir de response/style.",
    )
    build_profile_parser = subparsers.add_parser(
        "build-profile",
        help="Extrae un perfil estructurado de Wenceslao a partir de los chats.",
    )
    build_profile_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora cache local y fuerza reconstruccion del perfil.",
    )
    subparsers.add_parser(
        "profile-report",
        help="Muestra un resumen del perfil estructurado extraido.",
    )
    build_rag_parser = subparsers.add_parser(
        "build-rag",
        help="Ejecuta prepare-data, build-memory, build-profile y build-index.",
    )
    build_rag_parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza reconstruccion de memoria antes de indexar.",
    )
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Inspecciona la recuperacion dual para una consulta.",
    )
    retrieve_parser.add_argument("query", help="Consulta a recuperar.")
    retrieve_parser.add_argument("--k-response", type=int, default=6)
    retrieve_parser.add_argument("--k-style", type=int, default=8)
    retrieve_parser.add_argument("--min-score", type=float, default=0.22)
    answer_parser = subparsers.add_parser(
        "answer",
        help="Genera una respuesta final usando retrieval + OpenAI.",
    )
    answer_parser.add_argument("query", help="Consulta a responder.")
    answer_parser.add_argument("--k-total", type=int, default=10)
    answer_parser.add_argument("--min-score", type=float, default=0.22)
    answer_parser.add_argument("--temperature", type=float, default=None)
    answer_parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Muestra tambien los hits de response/style usados.",
    )
    chat_parser = subparsers.add_parser(
        "chat",
        help="Abre un bucle interactivo de chat sobre el indice RAG.",
    )
    chat_parser.add_argument("--k-total", type=int, default=10)
    chat_parser.add_argument("--min-score", type=float, default=0.22)
    chat_parser.add_argument("--temperature", type=float, default=None)
    return parser


def _run_ingest(parsed_args: argparse.Namespace) -> int:
    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    input_files = _ensure_existing_input_dir(config.paths.chats_dir)
    messages = load_messages_from_directory(
        config.paths.chats_dir,
        my_name=config.my_name,
    )
    if messages.empty:
        raise SystemExit(
            "Se encontraron exportes, pero no se pudo extraer ningun mensaje "
            "valido.\n"
            f"Ruta: {config.paths.chats_dir}\n"
            f"Ficheros detectados: {len(input_files)}\n"
            "Comprueba que sean exportes de WhatsApp compatibles o CSV con "
            "columnas `question` y `wenceslao_answer`."
        )

    target_path = config.paths.data_dir / "messages.parquet"
    saved_path = _save_dataframe(messages, target_path)
    print(
        f"[ingest] mensajes={len(messages)} | chats_dir={config.paths.chats_dir}"
    )
    print(f"[ingest] guardado en {saved_path}")
    return 0


def _run_clean(parsed_args: argparse.Namespace) -> int:
    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    source_messages = _load_required_artifact(
        config.paths.data_dir / "messages",
        build_hint="wharagbot ingest",
    )
    cleaned_messages, resolution = clean_messages(
        source_messages,
        configured_name=config.my_name,
    )
    if cleaned_messages.empty:
        raise SystemExit(
            "La limpieza ha producido 0 mensajes. "
            "Revisa `MY_NAME` y el contenido de los exportes."
        )
    target_path = config.paths.data_dir / "messages_clean.parquet"
    saved_path = _save_dataframe(cleaned_messages, target_path)
    print(
        f"[clean] mensajes={len(cleaned_messages)} | my_name={resolution.name!r} "
        f"| source={resolution.source}"
    )
    print(f"[clean] guardado en {saved_path}")
    return 0


def _run_prepare_data(parsed_args: argparse.Namespace) -> int:
    _run_ingest(parsed_args)
    return _run_clean(parsed_args)


def _run_build_memory(parsed_args: argparse.Namespace) -> int:
    from wharagbot.memory.builders import (
        MemoryBuildSettings,
        build_or_load_memory_artifacts,
    )

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    print("[build-memory] leyendo messages_clean...", flush=True)
    cleaned_messages = _load_required_artifact(
        config.paths.data_dir / "messages_clean",
        build_hint="wharagbot prepare-data",
    )
    try:
        print(
            f"[build-memory] construyendo unidades desde {len(cleaned_messages)} "
            "mensajes limpios...",
            flush=True,
        )
        result = build_or_load_memory_artifacts(
            cleaned_messages,
            data_dir=config.paths.data_dir,
            ingest_manifest_path=(
                config.paths.data_dir / "ingest_cache" / "manifest.json"
            ),
            ctx_window=config.ctx_window,
            my_name=config.my_name,
            settings=MemoryBuildSettings(),
            force=bool(getattr(parsed_args, "force", False)),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "[build-memory] "
        f"memory={len(result['memory_units'])} "
        f"response={len(result['response_units'])} "
        f"style={len(result['style_units'])} "
        f"fact={len(result['fact_units'])}"
    )
    print(
        "[build-memory] "
        f"memory_cache_hit={result['memory_cache_hit']} "
        f"dual_cache_hit={result['dual_cache_hit']}"
    )
    return 0


def _run_build_index(parsed_args: argparse.Namespace) -> int:
    from wharagbot.retrieval.indexing import (
        IndexBuildSettings,
        build_dual_indices,
    )

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    response_units = _load_required_artifact(
        config.paths.data_dir / "response_units",
        build_hint="wharagbot build-memory",
    )
    style_units = _load_required_artifact(
        config.paths.data_dir / "style_units",
        build_hint="wharagbot build-memory",
    )
    fact_units = _load_required_artifact(
        config.paths.data_dir / "fact_units",
        build_hint="wharagbot build-memory",
    )
    result = build_dual_indices(
        response_units,
        style_units,
        fact_units,
        index_dir=config.paths.index_dir,
        embed_model=config.embed_model,
        settings=IndexBuildSettings(
            verbose=True,
            show_progress_bar=True,
        ),
    )
    print(
        "[build-index] "
        f"response={len(result['response_units'])} "
        f"style={len(result['style_units'])} "
        f"fact={len(result['fact_units'])} "
        f"index_dir={config.paths.index_dir}"
    )
    return 0


def _run_build_profile(parsed_args: argparse.Namespace) -> int:
    from wharagbot.profile import (
        ProfileBuildSettings,
        build_or_load_profile_artifacts,
    )

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    print("[build-profile] leyendo messages_clean...", flush=True)
    cleaned_messages = _load_required_artifact(
        config.paths.data_dir / "messages_clean",
        build_hint="wharagbot prepare-data",
    )
    result = build_or_load_profile_artifacts(
        cleaned_messages,
        data_dir=config.paths.data_dir,
        my_name=config.my_name,
        force=bool(getattr(parsed_args, "force", False)),
        settings=ProfileBuildSettings(),
    )
    print(
        "[build-profile] "
        f"facts={len(result['profile_facts'])} "
        f"cache_hit={result['cache_hit']}"
    )
    print(
        "[build-profile] "
        f"guardado en {config.paths.data_dir / 'profile_facts.parquet'}"
    )
    return 0


def _run_profile_report(parsed_args: argparse.Namespace) -> int:
    from wharagbot.profile import PROFILE_SUMMARY_FILE_NAME
    from wharagbot.utils import load_json

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    summary_path = config.paths.data_dir / PROFILE_SUMMARY_FILE_NAME
    summary = load_json(summary_path, {})
    if not summary:
        raise SystemExit(
            "No existe resumen de perfil. Ejecuta `wharagbot build-profile` primero."
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _run_build_rag(parsed_args: argparse.Namespace) -> int:
    print("[build-rag] fase=prepare-data", flush=True)
    _run_prepare_data(parsed_args)
    print("[build-rag] fase=build-memory", flush=True)
    _run_build_memory(parsed_args)
    print("[build-rag] fase=build-profile", flush=True)
    _run_build_profile(parsed_args)
    print("[build-rag] fase=build-index", flush=True)
    result = _run_build_index(parsed_args)
    print("[build-rag] completado", flush=True)
    return result


def _run_retrieve(parsed_args: argparse.Namespace) -> int:
    from wharagbot.retrieval.search import (
        load_dual_index_bundle,
        retrieve_bundle,
    )

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    try:
        bundle = load_dual_index_bundle(
            index_dir=config.paths.index_dir,
            embed_model=config.embed_model,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    result = retrieve_bundle(
        bundle,
        query=parsed_args.query,
        k_response=parsed_args.k_response,
        k_style=parsed_args.k_style,
        min_score=parsed_args.min_score,
    )
    profile_facts = _load_optional_artifact(config.paths.data_dir / "profile_facts")
    if not profile_facts.empty:
        from wharagbot.profile.search import retrieve_profile_facts

        result["profile_hits"] = retrieve_profile_facts(
            profile_facts,
            query=parsed_args.query,
            k=max(parsed_args.k_style, parsed_args.k_response),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _run_answer(parsed_args: argparse.Namespace) -> int:
    from wharagbot.generation.chat import generate_answer
    from wharagbot.retrieval.search import load_dual_index_bundle

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    if not config.openai_api_key:
        raise SystemExit("Falta OPENAI_API_KEY en .env para usar `answer`.")

    try:
        bundle = load_dual_index_bundle(
            index_dir=config.paths.index_dir,
            embed_model=config.embed_model,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    profile_facts = _load_optional_artifact(config.paths.data_dir / "profile_facts")
    result = generate_answer(
        bundle=bundle,
        my_name=config.my_name,
        prompt=parsed_args.query,
        api_key=config.openai_api_key,
        gen_model=config.gen_model,
        k_total=parsed_args.k_total,
        min_score=parsed_args.min_score,
        temperature=parsed_args.temperature,
        profile_facts=profile_facts,
    )
    print(result["answer"])
    if parsed_args.show_hits:
        print(
            json.dumps(
                {
                    "profile_hits": result.get("profile_hits", []),
                    "response_hits": result["response_hits"],
                    "style_hits": result["style_hits"],
                    "fact_hits": result["fact_hits"],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    return 0


def _run_chat(parsed_args: argparse.Namespace) -> int:
    from wharagbot.generation.chat import generate_answer
    from wharagbot.retrieval.search import load_dual_index_bundle

    config = load_runtime_config(
        project_root=parsed_args.project_root,
        chats_dir=parsed_args.chats_dir,
        my_name=parsed_args.my_name,
    )
    if not config.openai_api_key:
        raise SystemExit("Falta OPENAI_API_KEY en .env para usar `chat`.")

    try:
        bundle = load_dual_index_bundle(
            index_dir=config.paths.index_dir,
            embed_model=config.embed_model,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    profile_facts = _load_optional_artifact(config.paths.data_dir / "profile_facts")
    print("Escribe tu mensaje. Usa `exit`, `quit` o `salir` para terminar.")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "salir"}:
            break
        result = generate_answer(
            bundle=bundle,
            my_name=config.my_name,
            prompt=query,
            api_key=config.openai_api_key,
            gen_model=config.gen_model,
            k_total=parsed_args.k_total,
            min_score=parsed_args.min_score,
            temperature=parsed_args.temperature,
            profile_facts=profile_facts,
        )
        print(result["answer"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed_args = parser.parse_args(argv)

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "ingest": _run_ingest,
        "clean": _run_clean,
        "prepare-data": _run_prepare_data,
        "build-memory": _run_build_memory,
        "build-profile": _run_build_profile,
        "build-index": _run_build_index,
        "profile-report": _run_profile_report,
        "build-rag": _run_build_rag,
        "retrieve": _run_retrieve,
        "answer": _run_answer,
        "chat": _run_chat,
    }
    return handlers[parsed_args.command](parsed_args)
