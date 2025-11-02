import os
import pandas as pd
from dashscope import Generation

print("✅ DashScope loaded:", Generation)
if not os.getenv("DASHSCOPE_API_KEY"):
    print("⚠️ Warning: DASHSCOPE_API_KEY is not set. Please run:")
    print("   setx DASHSCOPE_API_KEY your_api_key_here")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(base_dir, "data", "processed", "final_ai_attitudes.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

df = pd.read_csv(data_path)

try:
    platforms = ", ".join(df["platform"].dropna().unique())
except Exception:
    platforms = "unknown"

sentiment_counts = (
    df["sentiment_label"].value_counts(normalize=True).to_dict()
    if "sentiment_label" in df
    else {}
)
avg_toxicity = (
    round(df["toxicity"].astype(float).mean(), 3)
    if "toxicity" in df
    else "N/A"
)

context = f"""
📊 Project: Public Attitudes Toward AI
Platforms: {platforms}
Average toxicity: {avg_toxicity}
Sentiment distribution: {sentiment_counts}

Each record contains: platform, text, sentiment_label, and toxicity.
You can ask to compare platforms, describe results, or explain patterns.
"""
def ask_qwen(question: str) -> str:
    """Send a question + project context to Qwen Turbo."""
    prompt = f"""
You are a data analyst assistant helping with the project:
"Public Attitudes Toward AI".

Use the dataset summary below to answer questions clearly and concisely.

CONTEXT:
{context}

QUESTION:
{question}
"""

    try:
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        
        if isinstance(response, dict):
            
            if "output" in response and isinstance(response["output"], dict):
                text = response["output"].get("text") or response["output"].get("choices", [{}])[0].get("message", {}).get("content")
                if text:
                    return text
            elif "result" in response:
                return str(response["result"])

        elif hasattr(response, "output_text"):
            return response.output_text
        elif hasattr(response, "output"):
            return str(response.output)

        return str(response)

    except Exception as e:
        import traceback
        print("DashScope error traceback:", traceback.format_exc())
        return f"[Error calling Qwen API: {e}]"

def assistant_graph_qwen(user_input: str) -> str:
    """Entry point called from Streamlit dashboard."""
    return ask_qwen(user_input)
