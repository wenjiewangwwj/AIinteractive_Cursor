# Dual LLM Hub (Streamlit)

One place to talk to **OpenAI (ChatGPT)** and **Anthropic (Claude)** together: a single task, optional shared opinion, and shared attachments. Both models answer in parallel. If you enable **cross-review**, each model automatically receives the other’s round‑1 answer (no copy-paste). You can extend the flow in code if you want more than two rounds.

## Is this possible?

Yes. This app orchestrates two HTTP APIs from your machine (or Streamlit Cloud): your prompt and files are sent to each provider; round two injects the peer’s text into each call. It is **not** a native “ChatGPT chats with Claude” product feature—it is **your** app wiring two APIs.

## Requirements

- Python 3.10+
- API keys: [OpenAI](https://platform.openai.com/api-keys) and [Anthropic](https://console.anthropic.com/)

## Local setup

```bash
cd "path/to/Cursor Generate Code"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY and ANTHROPIC_API_KEY
streamlit run app.py
```

On macOS/Linux use `source .venv/bin/activate` instead of `.\.venv\Scripts\activate`.

## GitHub + Streamlit Community Cloud

1. Push this folder to a GitHub repo (do **not** commit `.env` or real keys).
2. On [Streamlit Cloud](https://streamlit.io/cloud), deploy the repo and set **Secrets** to:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

3. Main file: `app.py`.

## Attachments

- **Text**: `.txt`, `.md` (UTF-8) inlined into the prompt.
- **PDF**: text extracted with `pypdf` (scanned PDFs may be empty).
- **Word**: `.docx` text extracted with `python-docx` (paragraphs and tables; `.doc` is not supported).
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` sent to vision-capable models.

Very large text is truncated with a notice (see `attachments.py`).

## Customization

- **More rounds**: edit `dual_llm.py` and add further exchanges (each side can see updated peer text).
- **Different workflow**: e.g. only Claude sees OpenAI, or a moderator model—same pattern: build the next user message from prior outputs.

## Push to GitHub

From your own machine (with [Git](https://git-scm.com/downloads) installed and your GitHub account):

```bash
cd "path/to/Cursor Generate Code"
git init
git add app.py attachments.py dual_llm.py requirements.txt README.md .gitignore .env.example
git commit -m "Add Dual LLM Streamlit hub with docx attachments"
```

Create an empty repo on GitHub (no README), then:

```bash
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Use a [Personal Access Token](https://github.com/settings/tokens) as the password when HTTPS prompts you, or set up SSH keys.

## Security

Treat API keys like passwords. Prefer Streamlit secrets or environment variables; avoid committing keys to GitHub.

