import streamlit as st
import json, os
from config import DATA_DIR

TASK_FILE = os.path.join(DATA_DIR, "tasks.json")

def load_tasks():
    if os.path.exists(TASK_FILE):
        return json.load(open(TASK_FILE))
    return []

def save_tasks(tasks):
    json.dump(tasks, open(TASK_FILE, "w"), indent=2)

st.title("🏠 FlatMate AI Checklist")

tasks = load_tasks()

if not tasks:
    st.info("No tasks yet. Run the analyzer first.")
else:
    for i, t in enumerate(tasks):
        done = st.checkbox(f"{t['person']}: {t['task']}", value=t["status"] == "done")
        if done and t["status"] != "done":
            t["status"] = "done"
            t["completed"] = "now"
    save_tasks(tasks)
    st.success("Tasks updated.")

st.caption("Refresh after running a new analysis to see new items.")
