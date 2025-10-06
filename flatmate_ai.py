# flatmate_ai.py — FlatMate AI Assistant
# Summarizes chat-like messages and detects overall mood.

from transformers import pipeline
from datetime import datetime
from config import HUGGINGFACE_TOKEN, DATA_DIR
import os
import sys
import statistics

# ----------------------------------------------------------
# 1. Initialize pipelines
# ----------------------------------------------------------

# Chat/dialogue summarization model (SAMSum)
try:
    summarizer = pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum",
        token=HUGGINGFACE_TOKEN
    )
except TypeError:
    summarizer = pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum",
        use_auth_token=HUGGINGFACE_TOKEN
    )

# Sentiment model (multilingual BERT)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# ----------------------------------------------------------
# 2. Collect chat-style input
# ----------------------------------------------------------

print("\nFlatMate AI — Flat Communication Summarizer")
print("Paste chat messages between flatmates. Type 'done' on a new line to finish.\n")

lines = []
while True:
    msg = input()
    if msg.lower().strip() == "done":
        break
    if msg.strip():
        lines.append(msg.strip())

if not lines:
    print("No messages entered. Exiting.")
    sys.exit(1)

conversation = " ".join(lines)

# ----------------------------------------------------------
# 3. Summarize chat
# ----------------------------------------------------------

print("\nSummarizing conversation...")
summary_output = summarizer(
    conversation,
    max_length=120,
    min_length=25,
    do_sample=False
)
summary_text = summary_output[0]["summary_text"].strip()

# ----------------------------------------------------------
# 4. Sentiment analysis
# ----------------------------------------------------------

print("Analyzing sentiment...")
sentiments = sentiment_analyzer(lines)
scores = [s["label"] for s in sentiments]

# Convert labels like "4 stars" → numeric average
numeric_scores = []
for s in scores:
    try:
        numeric_scores.append(int(s.split()[0]))
    except Exception:
        pass

avg_score = statistics.mean(numeric_scores) if numeric_scores else 3
if avg_score >= 4:
    mood = "Positive"
elif avg_score <= 2:
    mood = "Negative"
else:
    mood = "Neutral"

# ----------------------------------------------------------
# 5. Save results
# ----------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)
filename = f"flatmate_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
file_path = os.path.join(DATA_DIR, filename)

with open(file_path, "w", encoding="utf-8") as f:
    f.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
    f.write("Messages:\n" + "\n".join(lines) + "\n\n")
    f.write("Summary:\n" + summary_text + "\n\n")
    f.write(f"Overall Mood: {mood} (Avg score {avg_score:.1f})\n")

# ----------------------------------------------------------
# 6. Display results
# ----------------------------------------------------------

print("\nSummary Generated Successfully.\n")
print("Summary:")
print("-" * 60)
print(summary_text)
print("-" * 60)
print(f"Mood: {mood} (Avg score {avg_score:.1f})")
print(f"\nSaved summary file: {file_path}\n")
