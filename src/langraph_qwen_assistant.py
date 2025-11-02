import os
import pandas as pd
from dashscope import Generation
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- Define conversation state ---
class ChatState(TypedDict):
    input: str
    context: str
    output: str


# --- Load dataset ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(base_dir, "data", "processed", "final_ai_attitudes.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = pd.DataFrame(columns=["platform", "text", "sentiment_label", "toxicity"])


# --- Qwen call helper ---
def qwen_chat(prompt: str) -> str:
    """Try calling Qwen API; fallback to local summary if it fails."""
    try:
        resp = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

        # Handle multiple possible formats from Qwen SDK
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text

        if hasattr(resp, "output") and resp.output:
            out = resp.output
            if isinstance(out, list) and "text" in out[0]:
                return out[0]["text"]
            if isinstance(out, dict) and "text" in out:
                return out["text"]
            return str(out)

        return "⚠️ No valid response content returned from Qwen."
    except Exception as e:
        return f"[LOCAL FALLBACK] Could not reach Qwen API ({e})"


# --- Node 1: Generate numeric summary ---
def summarize_node(state: ChatState):
    """Compute dataset-level summary as context."""
    if df.empty:
        summary_text = "No data available for summary."
    else:
        avg_tox = round(df["toxicity"].astype(float).mean(), 3)
        avg_sent = round(df["sentiment_score"].mean(), 3) if "sentiment_score" in df else "N/A"

        platform_summary = (
            df.groupby("platform")[["sentiment_score", "toxicity"]]
            .mean()
            .reset_index()
            .to_string(index=False)
        )

        summary_text = f"""
        Platforms analyzed:
        {platform_summary}

        Overall Average Sentiment: {avg_sent}
        Overall Average Toxicity: {avg_tox}
        """

    return {"context": summary_text}


# --- Node 2: Reasoning / Qwen or fallback ---
def llm_node(state: ChatState):
    """Use Qwen to reason; fallback to local text generation if Qwen fails."""
    question = state["input"]
    context = state.get("context", "")

    # Prepare context for Qwen
    prompt = f"""
    You are a data analyst assistant for a project analyzing public attitudes toward AI.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Answer concisely and clearly in 2–4 sentences, using numbers where possible.
    """

    # 1️⃣ Try Qwen
    answer = qwen_chat(prompt)

    # 2️⃣ If Qwen failed, use local fallback
    if "[LOCAL FALLBACK]" in answer or "Error" in answer or "⚠️" in answer:
        try:
            answer = local_fallback_answer(question)
        except Exception as e:
            answer = f"⚠️ Unable to generate a local summary ({e})."

    return {"output": answer}


# --- Local fallback generator ---
def local_fallback_answer(question: str) -> str:
    """Generate an offline answer using the dataframe."""
    if df.empty:
        return "No local data available to summarize."

    avg_tox = round(df["toxicity"].astype(float).mean(), 3)
    avg_sent = round(df["sentiment_score"].mean(), 3) if "sentiment_score" in df else "N/A"
    grouped = df.groupby("platform")[["sentiment_score", "toxicity"]].mean().reset_index()

    # Keyword-based logic for simple local summaries
    q_lower = question.lower()

    if "compare" in q_lower:
        return (
            f"Locally computed comparison:\n"
            f"{grouped.to_string(index=False)}\n"
            f"Average sentiment: {avg_sent}, toxicity: {avg_tox}."
        )

    elif "most positive" in q_lower or "best" in q_lower:
        top = grouped.loc[grouped["sentiment_score"].idxmax()]
        return f"The most positive platform is {top['platform']} (sentiment {round(top['sentiment_score'],3)})."

    elif "most negative" in q_lower or "worst" in q_lower:
        low = grouped.loc[grouped["sentiment_score"].idxmin()]
        return f"The most negative platform is {low['platform']} (sentiment {round(low['sentiment_score'],3)})."

    elif "toxic" in q_lower:
        high_tox = grouped.loc[grouped["toxicity"].idxmax()]
        return f"The most toxic platform is {high_tox['platform']} (toxicity {round(high_tox['toxicity'],3)})."

    else:
        return (
            f"Local summary: Average sentiment = {avg_sent}, Average toxicity = {avg_tox}.\n"
            "Platforms analyzed:\n" + grouped.to_string(index=False)
        )


# --- Build LangGraph pipeline ---
def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


assistant_graph_qwen = build_graph()


# --- Manual test mode ---
if __name__ == "__main__":
    user_q = input("Ask something: ")
    result = assistant_graph_qwen.invoke({"input": user_q})
    print("💬 Qwen Assistant →", result["output"])
