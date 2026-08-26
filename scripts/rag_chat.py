r"""Project Gorgon Gradio chat spike.

Gradio ChatInterface over pgrag's streaming pipeline (ask_stream).
Requires the embed (:8081) and LLM (:8080) services to be running.

Run: mise chat   (uv run --with gradio python scripts/rag_chat.py)
"""

import os

import gradio as gr

from pgrag.rag.pipeline import ask_stream


def respond(message, history):
    """Chat handler: streams tokens from ask_stream(), then appends sources.

    history is threaded by Gradio but the pipeline is one-shot (it only
    answers from the latest message), matching the Open WebUI pipe.
    """
    full = ""
    result = None
    try:
        for event in ask_stream(message):
            if event["type"] == "reset":
                full = ""
            elif event["type"] == "token":
                full += event["text"]
                yield full
            elif event["type"] == "final":
                result = event["result"]
    except Exception as exc:
        yield f"Sorry, something went wrong: {exc}"
        return

    if result is None:
        yield "Sorry, no answer was produced."
        return

    sources = result["sources"]
    source_block = "\n".join(
        f"- {s['citation']}" for s in sources
    ) or "No sources found."
    yield full + f"\n\n---\n**Sources:**\n{source_block}"


demo = gr.ChatInterface(
    respond,
    title="Project Gorgon RAG",
    description=(
        "Ask about Project Gorgon skills, recipes, items, quests, and "
        "abilities. Requires the embed and LLM services to be running."
    ),
    save_history=True,
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        # `mise start` (chat-start.ps1) sets PG_RAG_CHAT_NO_BROWSER=1 so the
        # serving path doesn't pop a tab; interactive `mise chat` still opens one.
        inbrowser=not os.environ.get("PG_RAG_CHAT_NO_BROWSER"),
    )
