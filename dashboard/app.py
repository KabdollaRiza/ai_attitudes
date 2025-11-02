import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Public Attitudes Toward AI",
    layout="wide",
)

@st.cache_data
def load_data():
    
    df = pd.read_csv("../data/processed/final_ai_attitudes.csv")

    rename_map = {
        "sentiment_label": "sentiment",
        "topic_id": "topic",
        "TOXICITY": "toxicity",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "platform" in df.columns:
        df["platform"] = (
            df["platform"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "reddit": "Reddit",
                "r/reddit": "Reddit",
                "reddit.com": "Reddit",
                "tw": "Twitter",
                "twitter": "Twitter",
                "x": "Twitter",
                "hackernews": "Hacker News",
                "hacker news": "Hacker News",
                "hn": "Hacker News",
                "ai": "Reddit",
            })
        )

    for col in ["sentiment_score", "toxicity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


df = load_data()

st.sidebar.header(" Filter Options")

platforms = sorted(df["platform"].dropna().unique())
selected_platforms = st.sidebar.multiselect(
    "Choose Platform(s)", platforms, default=platforms
)

filtered_df = df[df["platform"].isin(selected_platforms)]

st.title(" Public Attitudes Toward AI: Comparative Dashboard")
st.markdown(
    "Analyze public opinions about AI across **Reddit**, **Twitter (X)**, and **Hacker News**. "
    "Explore sentiment, emotions, topics, and toxicity levels interactively."
)

st.subheader(" Sentiment Distribution")
if "sentiment" in filtered_df.columns:
    fig_sent = px.histogram(
        filtered_df,
        x="sentiment",
        color="sentiment",
        title=f"Sentiment Distribution ({', '.join(selected_platforms)})",
        text_auto=True,
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.warning(" Sentiment data not available in this dataset.")

st.subheader(" Emotion Breakdown")
emotion_cols = [col for col in filtered_df.columns if col.startswith("emotion_")]

if emotion_cols:
    melted = filtered_df.melt(value_vars=emotion_cols, var_name="emotion", value_name="emotion_value")
    fig_emo = px.box(
        melted,
        x="emotion",
        y="emotion_value",
        title="Emotion Intensity by Category",
        color="emotion"
    )
    st.plotly_chart(fig_emo, use_container_width=True)
else:
    st.info("No emotion columns found in dataset.")

st.subheader(" Toxicity vs Sentiment Score")
if {"toxicity", "sentiment_score"}.issubset(filtered_df.columns):
    fig_tox = px.scatter(
        filtered_df,
        x="sentiment_score",
        y="toxicity",
        color="sentiment",
        hover_data=["text", "platform"],
        title=f"Toxicity vs Sentiment Score ({', '.join(selected_platforms)})",
    )
    st.plotly_chart(fig_tox, use_container_width=True)
else:
    st.warning(" Missing columns: 'toxicity' or 'sentiment_score'")

st.subheader(" Top Topics")

if "topic" in df.columns:
    if not filtered_df.empty:
        data_to_use = filtered_df
    else:
        data_to_use = df

    top_topics = (
        data_to_use["topic"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "topic", "topic": "count"})
    )

    if "count" not in top_topics.columns or "topic" not in top_topics.columns:
        top_topics.columns = ["topic", "count"]

    if not top_topics.empty:
        fig_topics = px.bar(
            top_topics.head(10),
            x="topic",
            y="count",
            title="Top 10 Topics by Frequency",
            text_auto=True,
        )
        st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.info("No topic data available.")
else:
    st.info("Topic column not found in dataset.")

st.markdown("---")
st.subheader(" Summary Metrics")

col1, col2, col3 = st.columns(3)
if not filtered_df.empty:
    col1.metric("Average Sentiment Score", f"{filtered_df['sentiment_score'].mean():.2f}")
    col2.metric("Average Toxicity", f"{filtered_df['toxicity'].mean():.2f}")
    col3.metric("Total Posts", len(filtered_df))
else:
    st.info("No data available for the selected platform(s).")

from src.chat.assistant_graph_qwen import assistant_graph_qwen

st.markdown("### Ask about this Project (Qwen Turbo Assistant)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about AI Attitudes (e.g., 'What is the sentiment on Reddit?')")

if user_input:
    st.chat_message("user").write(user_input)

    try:
        answer = assistant_graph_qwen(user_input)
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

for user, bot in st.session_state.chat_history:
    st.chat_message("user").write(user)
    st.chat_message("assistant").write(bot)
