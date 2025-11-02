from transformers import pipeline
from src.utils.io import read_csv, write_csv

INP = "data/clean/all_clean.csv"
OUT = "data/processed/with_sentiment.csv"

def main():
    print("🔍 Running sentiment analysis...")
    df = read_csv(INP)

    model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True
    )

    # ✅ Process in small batches to prevent freeze
    texts = df["text"].tolist()
    labels, scores = [], []
    batch_size = 8  # you can adjust (8–16 recommended)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            preds = model(batch, truncation=True, max_length=512)
            labels.extend([p["label"] for p in preds])
            scores.extend([p["score"] for p in preds])
        except Exception as e:
            print(f"⚠️ Error on batch {i}: {e}")
            labels.extend(["ERROR"] * len(batch))
            scores.extend([0.0] * len(batch))

        # progress tracker
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)}")

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    write_csv(df, OUT)
    print(f"✅ Sentiment results saved to {OUT}")

if __name__ == "__main__":
    main()
