"""
Streamlit hub: one prompt, one upload set, optional user opinion — ChatGPT and Claude
answer in parallel, then optionally cross-review each other's answers automatically.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from attachments import prepare_uploads
from dual_llm import run_dual_session

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, default)
        return v if isinstance(v, str) else default
    except Exception:
        return default


def main() -> None:
    st.set_page_config(page_title="Dual LLM Hub", layout="wide")
    st.title("Dual LLM Hub (ChatGPT + Claude)")
    st.caption(
        "Ask once. Attach once. Optionally add your opinion once. "
        "Both models respond; enable cross-review so each sees the other's answer without copy-paste."
    )

    with st.sidebar:
        st.subheader("API keys")
        st.markdown("Use sidebar inputs, `.env`, or [Streamlit secrets](https://docs.streamlit.io/develop/api-reference/connections/st.secrets_connection).")
        openai_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", "") or _secret("OPENAI_API_KEY"),
            type="password",
        )
        anthropic_key = st.text_input(
            "Anthropic API key",
            value=os.getenv("ANTHROPIC_API_KEY", "") or _secret("ANTHROPIC_API_KEY"),
            type="password",
        )
        st.subheader("Models")
        openai_model = st.selectbox(
            "OpenAI model",
            [
                "gpt-5.4-mini",
                "gpt-5.4-mini-2026-03-17",
                "gpt-5.4",
                "gpt-5.4-nano",
                "gpt-4o",
            ],
            index=0,
            help="IDs from OpenAI model docs; `gpt-5.4-mini` is the current default alias.",
        )
        anthropic_model = st.selectbox(
            "Anthropic model",
            [
                "claude-sonnet-4-6",
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
            ],
            index=0,
            help="`claude-sonnet-4-6` is the Sonnet 4.6 API id; older ids kept as fallbacks.",
        )
        cross_review = st.checkbox("Cross-review round (each sees the other's answer)", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        task = st.text_area(
            "Task or question (shared with both)",
            height=180,
            placeholder="Example: Rank these submissions with criteria ... or any open-ended task.",
        )
    with col_b:
        user_opinion = st.text_area(
            "Your opinion / constraints (optional, shared with both)",
            height=180,
            placeholder="Example: I value originality over polish; do not penalize non-native English.",
        )

    uploads = st.file_uploader(
        "Attachments (shared with both)",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "docx", "png", "jpg", "jpeg", "gif", "webp"],
    )

    run = st.button("Run", type="primary", use_container_width=True)

    if not run:
        return

    if not openai_key.strip() or not anthropic_key.strip():
        st.error("Please provide both API keys (sidebar, `.env`, or Streamlit secrets).")
        return
    if not task.strip():
        st.error("Please enter a task or question.")
        return

    ctx = prepare_uploads(list(uploads) if uploads else None)

    with st.spinner("Calling both models…"):
        result = run_dual_session(
            openai_key.strip(),
            anthropic_key.strip(),
            openai_model,
            anthropic_model,
            task.strip(),
            user_opinion.strip(),
            ctx,
            cross_review=cross_review,
        )

    st.divider()
    st.subheader("Round 1 — independent answers")
    c1, c2 = st.columns(2)
    r1o = result["round1"]["openai"]
    r1a = result["round1"]["anthropic"]
    with c1:
        st.markdown(f"**{r1o.provider}** (`{r1o.model}`)")
        if r1o.error:
            st.error(r1o.error)
        else:
            st.markdown(r1o.text)
    with c2:
        st.markdown(f"**{r1a.provider}** (`{r1a.model}`)")
        if r1a.error:
            st.error(r1a.error)
        else:
            st.markdown(r1a.text)

    if result.get("round2_note"):
        st.warning(result["round2_note"])

    r2 = result.get("round2")
    if r2:
        st.divider()
        st.subheader("Round 2 — after seeing each other")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**{r2['openai'].provider}** (`{r2['openai'].model}`)")
            if r2["openai"].error:
                st.error(r2["openai"].error)
            else:
                st.markdown(r2["openai"].text)
        with d2:
            st.markdown(f"**{r2['anthropic'].provider}** (`{r2['anthropic'].model}`)")
            if r2["anthropic"].error:
                st.error(r2["anthropic"].error)
            else:
                st.markdown(r2["anthropic"].text)


if __name__ == "__main__":
    main()
