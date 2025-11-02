# Public Attitudes Toward AI — Comparative Sentiment Dashboard Overview

This project analyzes public opinions about Artificial Intelligence (AI) collected from multiple online platforms (Reddit, Twitter, and Hacker News).
It provides an interactive Streamlit dashboard that visualizes sentiment, emotion, topic, and toxicity levels across sources.

A built-in AI Assistant (powered by Qwen-Turbo) allows users to ask natural-language questions about the dataset — for example:

“Which platform is the most positive?”
“Compare Reddit and Twitter sentiment.”
“What is the average toxicity?”

# Project Architecture
`graph TD
A[Data Acquisition Layer] --> B[Preprocessing & Cleaning]
B --> C[Sentiment & Emotion Analysis]
C --> D[Data Integration & Export]
D --> E[Interactive Streamlit Dashboard]
E --> F[Qwen Turbo Chat Assistant]`


# Features
1. Sentiment Analysis

Sentiment labels (positive, neutral, negative) generated per platform

Uses pre-trained CardiffNLP RoBERTa Sentiment Model

2. Emotion Classification

Classifies posts into emotion categories (e.g., joy, anger, fear, surprise)

Implemented in emotions.py

3. Toxicity Estimation

Measures the civility of conversations using a toxicity score

Visualized via Scatter Plot (Toxicity vs Sentiment)

4. Topic Modeling

Groups posts into meaningful AI-related discussion topics (e.g., automation, bias, jobs)

5. Interactive Chat Assistant (Qwen-Turbo)

The Qwen Turbo Assistant acts as a built-in AI analyst.
It uses DashScope API to interpret dataset summaries and provide natural language answers to user questions.

Capabilities:

Compare platforms by sentiment or toxicity

Explain which platform is most positive/negative

Discuss overall trends and dataset summary

Respond conversationally in natural language

# Qwen Assistant Logic
How it works:

Loads final_ai_attitudes.csv from /data/processed

Cleans and standardizes platform names (ai → reddit)

Automatically computes per-platform:

Sentiment proportions

Average toxicity

Builds a context prompt injected into Qwen-Turbo

When a user asks a question, it calls:

from dashscope import Generation
Generation.call(model="qwen-turbo", prompt=prompt)


Qwen interprets context + question → returns natural analytical answers

# How to Run
1. Install Dependencies
bash
`pip install -r requirements.txt`

3. Set up DashScope API Key

Get your API key from: https://dashscope.aliyun.com

Then set it in your environment:

On Windows PowerShell:
`setx DASHSCOPE_API_KEY "your_actual_api_key_here"`

On macOS/Linux:
`export DASHSCOPE_API_KEY="your_actual_api_key_here"`

3. Launch the Streamlit App
bash
`cd dashboard`
`streamlit run app.py`

# Dashboard Overview
Section	Description
Sentiment Distribution	Bar chart comparing positive, neutral, and negative shares
Emotion Breakdown	Pie or stacked chart of emotion categories (if available)
Toxicity vs Sentiment	Scatter plot showing relationship between toxicity and sentiment
Top Topics	Bar chart of most discussed AI-related topics
Qwen Turbo Assistant	Interactive chat to explore and interpret results
# Example Queries (Try These!)
Query	Example Answer
“Which platform is most positive?”	Hacker News has the highest positive sentiment and lowest toxicity.
“Compare Reddit and Twitter.”	Reddit is more balanced, while Twitter shows higher toxicity.
“What’s the average toxicity?”	The average toxicity across all platforms is 0.01.
“What is the overall sentiment?”	Most discussions are neutral, showing balanced public opinion.
# Tech Stack
Category	Tools / Frameworks
Language	Python 3.11
Frontend	Streamlit
AI Model	Qwen-Turbo via DashScope API
Libraries	pandas, plotly, transformers, sklearn
Data	Reddit, Twitter (X), Hacker News AI discussions
### Sample Output (Qwen Context Snippet)
Detailed platform-level statistics:
• Reddit: positive=0.48, neutral=0.42, negative=0.10, avg_toxicity=0.013
• Twitter: positive=0.45, neutral=0.38, negative=0.17, avg_toxicity=0.025
• Hacker News: positive=0.57, neutral=0.30, negative=0.13, avg_toxicity=0.009

# Insights

Hacker News tends to have the most optimistic discussions about AI.

Twitter has the highest proportion of negative posts and slightly higher toxicity.

Reddit is more balanced and diverse in sentiment topics.

# Authors
Riza Kabdolla, Ardak Islam, Kamila Nurlybayeva
