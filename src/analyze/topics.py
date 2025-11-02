from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from src.utils.io import read_csv, write_csv

INP = "data/processed/with_emotions.csv"
OUT = "data/processed/with_topics.csv"

def main():
    df = read_csv(INP)
    if df.empty:
        print("⚠️ No data found.")
        return

    print("🧠 Running BERTopic topic modeling...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=emb_model, verbose=True)
    topics, probs = topic_model.fit_transform(df["text"].tolist())

    df["topic_id"] = topics
    topic_model.get_topic_info().to_csv("data/processed/topic_info.csv", index=False)

    write_csv(df, OUT)
    print(f"✅ Topics saved to {OUT}")

if __name__ == "__main__":
    main()
