import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from src.utils.io import read_csv, write_csv

# Input/Output paths
INP = "data/processed/with_sentiment.csv"
OUT = "data/processed/with_emotions.csv"

def main():
    print("🎭 Running emotion analysis...")
    df = read_csv(INP)

    # Load emotion model
    emo_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

    # Analyze emotions for each text
    emotion_results = []
    for text in tqdm(df["text"].tolist(), desc="Predicting emotions"):
        try:
            scores = emo_model(text[:512])[0]  # truncate to avoid token limit
            emotion_results.append({e["label"].lower(): e["score"] for e in scores})
        except Exception as e:
            print(f"⚠️ Error on text: {e}")
            emotion_results.append({})

    # Convert results into a DataFrame
    emo_df = pd.DataFrame(emotion_results).fillna(0)

    # Merge back into main dataset
    df = pd.concat([df, emo_df.add_prefix("emotion_")], axis=1)

    write_csv(df, OUT)
    print(f"✅ Emotions saved to {OUT}")

if __name__ == "__main__":
    main()
