# WhaRAGBot
RAG sobre tus chats de WhatsApp para chatear "contigo mismo" usando OpenAI API.

## Estructura
- `Chats en .zip/` — coloca aquí los ZIP exportados de WhatsApp (sin multimedia). Ignorado por git.
- `notebooks/wha-ragbot.ipynb` — cuaderno paso a paso: parseo → ejemplos → embeddings → FAISS → chat.
- `data/`, `index/` — salidas intermedias (procesados, índices). Ignorados por git.

## Requisitos rápidos
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar el notebook
```bash
jupyter notebook notebooks/wha-ragbot.ipynb
```
En la sección de configuración, ajusta `CHATS_ZIP_DIR` y `MY_NAME` si tu nombre aparece distinto en los chats.

## Modelo generativo (OpenAI API)
Configura tus variables en `.env` antes de ejecutar la parte de chat:
```bash
cp .env.example .env
# edita .env y añade tu OPENAI_API_KEY real
```

## Notas
- Este repo es público: no subas `Chats en .zip/`, `data/` ni `index/`.
- Los embeddings usan `intfloat/multilingual-e5-small` por defecto; cambia `EMBED_MODEL` si necesitas algo más ligero.
- El pipeline asume exportes de WhatsApp en español; si algún chat no parsea, revisa `START_PATTERNS` en el notebook para adaptarlo.
