import pandas as pd
import os

def load_clean_file(path, platform_name):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["platform"] = platform_name  # ✅ set proper platform name
        print(f"Loaded {path}: {len(df)} rows")
        return df
    else:
        print(f"⚠️ Missing file: {os.path.basename(path)}")
        return pd.DataFrame()

def main():
    print("🔄 Merging clean datasets...")

    reddit = load_clean_file("data/clean/ai_reddit_posts.csv", "Reddit")
    twitter = load_clean_file("data/clean/twitter_ai_data_clean.csv", "Twitter")
    hackernews = load_clean_file("data/clean/hackernews_ai_comments.csv", "Hacker News")

    all_df = pd.concat([reddit, twitter, hackernews], ignore_index=True)
    out_path = "data/clean/all_clean.csv"
    all_df.to_csv(out_path, index=False)
    print(f"✅ Merged data saved to {out_path}")

if __name__ == "__main__":
    main()
