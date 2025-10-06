# app.py — FlatMate AI (Chat-Style Summarizer)
# Uses SAMSum fine-tuned model for realistic conversational summaries.

from transformers import pipeline
from datetime import datetime
from config import HUGGINGFACE_TOKEN, DATA_DIR
import os
import sys

# ----------------------------------------------------------
# 1. Load the SAMSum model
# ----------------------------------------------------------
# Why this model: it's trained on real human dialogues,
# making it ideal for daily notes, diary logs, or personal conversations.

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

# ----------------------------------------------------------
# 2. Accept user input
# ----------------------------------------------------------

print("\nFlatMate AI — Daily Summary Generator")
print("Enter your diary, meeting note, or chat text below.\n")

user_text = input("Enter text: ").strip()

if not user_text:
    print("Error: No text entered. Please run again.")
    sys.exit(1)

# ----------------------------------------------------------
# 3. Prepare safe summarization parameters
# ----------------------------------------------------------

word_count = len(user_text.split())

# Adjust automatically so even very short text works.
max_len = min(120, max(30, int(word_count * 1.8)))
min_len = max(10, int(max_len / 3))

print(f"\nSummarizing with model 'philschmid/bart-large-cnn-samsum' (max_length={max_len}, min_length={min_len})\n")

# ----------------------------------------------------------
# 4. Generate summary
# ----------------------------------------------------------

try:
    summary_output = summarizer(
        user_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    summary_text = summary_output[0]["summary_text"].strip()
except Exception as e:
    print(f"Error while summarizing: {e}")
    sys.exit(1)

# ----------------------------------------------------------
# 5. Save summary to local file
# ----------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

today = datetime.now().strftime("%Y%m%d")
file_path = os.path.join(DATA_DIR, f"summary_{today}.txt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
    f.write("Original Text:\n" + user_text + "\n\n")
    f.write("Summary:\n" + summary_text + "\n")

# ----------------------------------------------------------
# 6. Display output to user
# ----------------------------------------------------------

print("Summary Generated Successfully.\n")
print("Summary:")
print("-" * 60)
print(summary_text)
print("-" * 60)
print(f"\nSaved summary file: {file_path}\n")



