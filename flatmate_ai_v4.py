# flatmate_ai_v4.py — FlatMate AI (Summarizer + Task Tracker + Sentiment Fix)

from transformers import pipeline
from datetime import datetime
from config import HUGGINGFACE_TOKEN, DATA_DIR
import os, re, json, sys, schedule, time

# ----------------------------------------------------------------------
# 1. Initialize models
# ----------------------------------------------------------------------
print("Initializing FlatMate AI models...")

# Summarization: chat-style (SAMSum)
summarizer = pipeline(
    "summarization",
    model="philschmid/bart-large-cnn-samsum",
    token=HUGGINGFACE_TOKEN
)

# Sentiment: social/conversational model (less harsh)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# ----------------------------------------------------------------------
# 2. Task file utilities
# ----------------------------------------------------------------------
TASK_FILE = os.path.join(DATA_DIR, "tasks.json")

def load_tasks():
    if not os.path.exists(TASK_FILE):
        return []
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_tasks(tasks):
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

def add_tasks(new_tasks):
    tasks = load_tasks()
    tasks.extend(new_tasks)
    save_tasks(tasks)

def mark_done(person, keyword):
    tasks = load_tasks()
    updated = False
    for t in tasks:
        if (
            t["status"] == "pending"
            and t["person"].lower() == person.lower()
            and keyword.lower() in t["task"].lower()
        ):
            t["status"] = "done"
            t["completed"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            updated = True
    save_tasks(tasks)
    return updated

# ----------------------------------------------------------------------
# 3. Reminder engine
# ----------------------------------------------------------------------
def send_reminders():
    tasks = load_tasks()
    pending = [t for t in tasks if t["status"] == "pending"]
    if pending:
        print("\n🔔 Pending tasks:")
        for t in pending:
            print(f"- {t['person']}: {t['task']} (added {t['created']})")
        print("")
    else:
        print("\nAll tasks complete.\n")

# ----------------------------------------------------------------------
# 4. Extraction helpers
# ----------------------------------------------------------------------
def extract_tasks_from_chat(messages):
    """
    Extracts <Person> will/can/to <verb> directly from chat lines.
    """
    verbs = [
        "pay", "buy", "clean", "cook", "fix", "call", "collect", "arrange",
        "handle", "book", "water", "organize", "manage", "repair",
        "wash", "prepare", "take care", "bring"
    ]
    task_list = []
    for msg in messages:
        # Name detection improved: "A:", "A -", "A said:", etc.
        match = re.match(r"^([A-Z][a-zA-Z]*)[:\-]?\s*(.*)", msg)
        if match:
            person, content = match.groups()
        else:
            person, content = "Unknown", msg

        content = content.replace("I'll", "I will").replace("i'll", "i will")

        for verb in verbs:
            if re.search(rf"\b(will|can|to)\s{verb}\b", content.lower()):
                after = re.split(rf"{verb}", content, maxsplit=1, flags=re.IGNORECASE)
                task_text = verb + (after[1].strip() if len(after) > 1 else "")
                task_list.append({
                    "person": person,
                    "task": task_text,
                    "status": "pending",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                break
    return task_list

def extract_tasks_from_summary(summary_text):
    """
    Backup extractor from summarized text (regex + fallback).
    """
    patterns = [
        r"(\b[A-Z][a-z]+)\s(?:will|to|can|promised to|agreed to)\s([^\.]+)"
    ]
    tasks = []
    for pattern in patterns:
        for person, task in re.findall(pattern, summary_text):
            tasks.append({
                "person": person.strip(),
                "task": task.strip(),
                "status": "pending",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
    return tasks

# ----------------------------------------------------------------------
# 5. Conversation analyzer
# ----------------------------------------------------------------------
def analyze_conversation():
    print("\nFlatMate AI — Chat Summarizer & Task Tracker")
    print("Paste chat messages (type 'done' when finished):\n")

    lines = []
    while True:
        msg = input()
        if msg.lower().strip() == "done":
            break
        if msg.strip():
            lines.append(msg.strip())

    if not lines:
        print("No input provided.")
        sys.exit(0)

    conversation = " ".join(lines)

    # --- summarization ---
    print("\nSummarizing conversation...")
    summary = summarizer(conversation, max_length=120, min_length=30, do_sample=False)[0]["summary_text"].strip()

    # --- sentiment ---
    print("Analyzing sentiment...")
    result = sentiment_analyzer(conversation)[0]
    label = result["label"].upper()
    score = result["score"]
    if label in ("NEGATIVE", "LABEL_0") and score < 0.8:
        mood = "Neutral"
    elif label in ("NEGATIVE", "LABEL_0"):
        mood = "Negative"
    elif label in ("POSITIVE", "LABEL_2"):
        mood = "Positive"
    else:
        mood = "Neutral"

    print("\nSummary:\n", summary)
    print("Overall mood:", mood)

    # --- task extraction ---
    raw_tasks = extract_tasks_from_chat(lines)
    summary_tasks = extract_tasks_from_summary(summary)
    tasks = raw_tasks if raw_tasks else summary_tasks

    if tasks:
        print("\nTasks identified:")
        for t in tasks:
            print(f"- {t['person']}: {t['task']}")
        add_tasks(tasks)
        print(f"\n{len(tasks)} new task(s) added.")
    else:
        print("\nNo clear tasks found.")

    # --- save summary file ---
    filename = f"flatmate_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
    filepath = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n\n")
        f.write("Messages:\n" + "\n".join(lines) + "\n\n")
        f.write("Summary:\n" + summary + "\n\n")
        f.write("Mood: " + mood + "\n")

    print(f"\nSaved summary file: {filepath}\n")

# ----------------------------------------------------------------------
# 6. Run modes
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--done":
        if len(sys.argv) < 4:
            print("Usage: python flatmate_ai_v4.py --done <Person> <Keyword>")
        else:
            person, keyword = sys.argv[2], sys.argv[3]
            if mark_done(person, keyword):
                print(f"Marked '{keyword}' for {person} as done.")
            else:
                print("No matching task found.")
        sys.exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] == "--remind":
        print("Starting reminder service (every 6 hours)...")
        schedule.every(6).hours.do(send_reminders)
        send_reminders()
        while True:
            schedule.run_pending()
            time.sleep(60)

    else:
        analyze_conversation()
