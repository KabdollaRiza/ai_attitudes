from googleapiclient.discovery import build
from src.utils.io import read_csv, write_csv
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

INP = "data/processed/with_topics.csv"
OUT = "data/processed/final_ai_attitudes.csv"

def perspective_score(texts, attr="TOXICITY"):
    api_key = os.getenv("PERSPECTIVE_API_KEY")
    if not api_key:
        print("⚠️ Missing Perspective API key.")
        return [None] * len(texts)

    service = build(
        "commentanalyzer", "v1alpha1", developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
    )
    scores = []
    for t in texts[:500]:  # limit for free quota
        try:
            req = {'comment': {'text': t[:3000]},
                   'requestedAttributes': {attr: {}}}
            res = service.comments().analyze(body=req).execute()
            score = res["attributeScores"][attr]["summaryScore"]["value"]
        except Exception:
            score = None
        scores.append(score)
    return scores + [None] * (len(texts) - len(scores))

def main():
    df = read_csv(INP)
    if df.empty:
        print("⚠️ No data found.")
        return

    print("☢️ Running toxicity analysis...")
    df["toxicity"] = perspective_score(df["text"].tolist(), "TOXICITY")

    write_csv(df, OUT)
    print(f"✅ Final dataset saved to {OUT}")

if __name__ == "__main__":
    main()
