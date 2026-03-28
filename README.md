# WhaRAGBot
RAG sobre tus chats de WhatsApp para chatear "contigo mismo" usando OpenAI API.

## Estructura
- `Chats en .zip/` — coloca aquí los ZIP exportados de WhatsApp (sin multimedia). Ignorado por git.
- `notebooks/wha-ragbot.ipynb` — wrapper fino sobre la librería. Ya no contiene la lógica del pipeline.
- `src/wharagbot/` — implementación en `.py` del pipeline completo: config, ingesta, limpieza, memoria, embeddings, retrieval y generación.
- `tests/` — pruebas base para parseo y limpieza.
- `data/`, `index/` — salidas intermedias (procesados, índices). Ignorados por git.

## Requisitos rápidos
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Protección de secretos (pre-commit)
Activa hooks locales para bloquear claves y notebooks con outputs:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
El hook `repo-safety-check` bloquea commits con:
- claves API/tokens/privadas en texto
- archivos `.env*` (excepto `.env.example`)
- `AGENTS.md` / `agents.md`
- notebooks `.ipynb` con `outputs`

## CLI base
El pipeline base ya se puede ejecutar fuera del notebook:
```bash
source .venv/bin/activate
wharagbot build-rag
```

También puedes lanzar pasos por separado:
```bash
wharagbot ingest
wharagbot clean
wharagbot build-memory
wharagbot build-index
wharagbot retrieve "¿Vamos al cine?"
wharagbot answer "¿Vamos al cine?"
wharagbot chat
```

Flujo recomendado:
```bash
source .venv/bin/activate
wharagbot build-rag
wharagbot answer "¿Qué tal?"
```

Modo interactivo:
```bash
source .venv/bin/activate
wharagbot chat
```

## Notebook
El notebook sigue en el repo como wrapper y punto de inspección rápida, pero la ruta recomendada para usar y evolucionar la demo RAG ya es la implementación en `src/wharagbot/`.

## Modelo generativo (OpenAI API)
Configura tus variables en `.env` antes de ejecutar la parte de chat:
```bash
cp .env.example .env
# edita .env y añade tu OPENAI_API_KEY real
```

## Notas
- Este repo es público: no subas `Chats en .zip/`, `data/` ni `index/`.
- Los embeddings usan `intfloat/multilingual-e5-small` por defecto; cambia `EMBED_MODEL` si necesitas algo más ligero.
- El pipeline asume exportes de WhatsApp en español; si algún chat no parsea, revisa `START_PATTERNS` en [src/wharagbot/ingest/whatsapp.py](/Users/wensicm/Repositorios/WhaRAGBot/src/wharagbot/ingest/whatsapp.py).
