"""Call OpenAI and Anthropic with shared context and optional cross-review round."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from attachments import PreparedContext


@dataclass
class ModelReply:
    provider: str
    model: str
    text: str
    error: str | None = None


def _openai_user_content(prompt: str, ctx: PreparedContext) -> str | list[dict[str, Any]]:
    if not ctx.images:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for media_type, raw in ctx.images:
        b64 = base64.standard_b64encode(raw).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }
        )
    return parts


def _anthropic_user_blocks(prompt: str, ctx: PreparedContext) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for media_type, raw in ctx.images:
        b64 = base64.standard_b64encode(raw).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        )
    return blocks


def _extract_openai_text(resp: Any) -> str:
    msg = resp.choices[0].message
    return (msg.content or "").strip()


def _extract_anthropic_text(resp: Any) -> str:
    parts: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def call_openai(
    client: OpenAI,
    model: str,
    system: str,
    user_prompt: str,
    ctx: PreparedContext,
) -> str:
    content = _openai_user_content(user_prompt, ctx)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    return _extract_openai_text(resp)


def call_anthropic(
    client: Anthropic,
    model: str,
    system: str,
    user_prompt: str,
    ctx: PreparedContext,
) -> str:
    blocks = _anthropic_user_blocks(user_prompt, ctx)
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": blocks}],
    )
    return _extract_anthropic_text(resp)


def _round1_user_message(task: str, user_opinion: str, ctx: PreparedContext) -> str:
    parts = [
        "Shared task / question:",
        task.strip(),
    ]
    if ctx.text_block:
        parts += ["", "Context from attachments:", ctx.text_block]
    if user_opinion.strip():
        parts += [
            "",
            "User's own opinion or constraints (consider seriously):",
            user_opinion.strip(),
        ]
    parts += [
        "",
        "Answer independently and clearly. Another assistant will answer the same task; "
        "you may disagree. In a later step you might see their reasoning and refine your answer.",
    ]
    return "\n".join(parts)


def _round2_user_message(
    task: str,
    user_opinion: str,
    ctx: PreparedContext,
    self_label: str,
    self_prev: str,
    other_label: str,
    other_prev: str,
) -> str:
    parts = [
        "Original task / question:",
        task.strip(),
    ]
    if ctx.text_block:
        parts += ["", "Context from attachments:", ctx.text_block]
    if user_opinion.strip():
        parts += [
            "",
            "User's opinion (still applies):",
            user_opinion.strip(),
        ]
    parts += [
        "",
        f"Your earlier answer ({self_label}):",
        self_prev,
        "",
        f"The other assistant ({other_label}) answered:",
        other_prev,
        "",
        "Review both. You may keep your stance, merge ideas, or change your mind. "
        "Give a concise final answer and briefly note what you adopted or rejected from the other side.",
    ]
    return "\n".join(parts)


ROUND1_SYSTEM = (
    "You are a careful assistant. Follow the user's instructions. "
    "Be explicit when you are uncertain."
)

ROUND2_SYSTEM = (
    "You are collaborating in a two-model workflow. Be fair: weigh the other assistant's "
    "arguments on merit, not authority. Stay grounded in the user's task and attachments."
)


def run_dual_session(
    openai_key: str,
    anthropic_key: str,
    openai_model: str,
    anthropic_model: str,
    task: str,
    user_opinion: str,
    ctx: PreparedContext,
    cross_review: bool,
) -> dict[str, Any]:
    oa = OpenAI(api_key=openai_key)
    an = Anthropic(api_key=anthropic_key)

    u1 = _round1_user_message(task, user_opinion, ctx)

    r_openai: ModelReply
    r_claude: ModelReply

    def _safe_openai() -> ModelReply:
        try:
            t = call_openai(oa, openai_model, ROUND1_SYSTEM, u1, ctx)
            return ModelReply("OpenAI", openai_model, t, None)
        except Exception as e:  # noqa: BLE001
            return ModelReply("OpenAI", openai_model, "", str(e))

    def _safe_anthropic() -> ModelReply:
        try:
            t = call_anthropic(an, anthropic_model, ROUND1_SYSTEM, u1, ctx)
            return ModelReply("Anthropic", anthropic_model, t, None)
        except Exception as e:  # noqa: BLE001
            return ModelReply("Anthropic", anthropic_model, "", str(e))

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_safe_openai): "openai", ex.submit(_safe_anthropic): "anthropic"}
        results: dict[str, ModelReply] = {}
        for fut in as_completed(futs):
            key = futs[fut]
            results[key] = fut.result()
    r_openai = results["openai"]
    r_claude = results["anthropic"]

    out: dict[str, Any] = {
        "round1": {"openai": r_openai, "anthropic": r_claude},
        "round2": None,
    }

    if not cross_review:
        return out

    if r_openai.error or r_claude.error:
        out["round2_note"] = "Skipped cross-review because a round-1 call failed."
        return out

    u2_openai = _round2_user_message(
        task,
        user_opinion,
        ctx,
        "ChatGPT / OpenAI",
        r_openai.text,
        "Claude / Anthropic",
        r_claude.text,
    )
    u2_claude = _round2_user_message(
        task,
        user_opinion,
        ctx,
        "Claude / Anthropic",
        r_claude.text,
        "ChatGPT / OpenAI",
        r_openai.text,
    )

    def _r2_openai() -> ModelReply:
        try:
            t = call_openai(oa, openai_model, ROUND2_SYSTEM, u2_openai, ctx)
            return ModelReply("OpenAI", openai_model, t, None)
        except Exception as e:  # noqa: BLE001
            return ModelReply("OpenAI", openai_model, "", str(e))

    def _r2_claude() -> ModelReply:
        try:
            t = call_anthropic(an, anthropic_model, ROUND2_SYSTEM, u2_claude, ctx)
            return ModelReply("Anthropic", anthropic_model, t, None)
        except Exception as e:  # noqa: BLE001
            return ModelReply("Anthropic", anthropic_model, "", str(e))

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_r2_openai): "openai", ex.submit(_r2_claude): "anthropic"}
        r2: dict[str, ModelReply] = {}
        for fut in as_completed(futs):
            r2[futs[fut]] = fut.result()

    out["round2"] = {"openai": r2["openai"], "anthropic": r2["anthropic"]}
    return out
