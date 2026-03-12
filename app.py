import streamlit as st
import torch
import pandas as pd
import sqlite3
import time
import os
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- PERFORMANCE ----------------
torch.set_num_threads(1)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="GenSenti",
    page_icon="🧠",
    layout="wide"
)

LABELS = ["sadness", "anxiety", "emotional_fatigue", "fear", "joy", "neutral"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../gensenti_model")
DB_PATH = os.path.join(BASE_DIR, "gensenti.db")

# ---------------- CUSTOM UI THEME ----------------
st.markdown("""
<style>

body {
    margin:0;
    padding:0;
    background: linear-gradient(-45deg,#000000,#330033,#800080,#ff00cc);
    background-size:400% 400%;
    animation:gradientBG 12s ease infinite;
    color:white;
}

@keyframes gradientBG {
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

h1,h2,h3{
color:#ff66cc;
text-align:center;
}

.stButton>button{
background:linear-gradient(90deg,#ff00cc,#8000ff);
color:white;
border-radius:25px;
height:45px;
width:180px;
font-weight:bold;
border:none;
}

.stTextArea textarea{
background-color:#1a1a1a !important;
color:white !important;
border-radius:12px;
border:1px solid #ff00cc;
}

.stProgress > div > div > div > div{
background:linear-gradient(90deg,#ff00cc,#8000ff);
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
text TEXT,
timestamp TEXT,
sadness REAL,
anxiety REAL,
emotional_fatigue REAL,
fear REAL,
joy REAL,
neutral REAL,
interpretation TEXT,
suggestion TEXT
)
""")

conn.commit()

# ---------------- MODEL LOAD ----------------
@st.cache_resource(show_spinner=False)
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ---------------- FUNCTIONS ----------------
def generate_explanation(text, scores):

    primary = max(scores, key=scores.get)
    intensity = scores[primary]

    if intensity > 0.75:
        level = "high"
    elif intensity > 0.40:
        level = "moderate"
    else:
        level = "mild"

    return f"The text mainly expresses {primary} with {level} intensity."


def generate_advice(scores):

    primary = max(scores, key=scores.get)

    advice_bank = {

        "sadness":[
        "Consider talking to someone you trust.",
        "Writing your thoughts in a journal can help.",
        "Take a short walk to refresh your mind.",
        "Allow yourself time to process emotions."
        ],

        "anxiety":[
        "Try slow breathing for a few minutes.",
        "Focus on what you can control now.",
        "Break tasks into smaller steps.",
        "Step away from screens briefly."
        ],

        "emotional_fatigue":[
        "Consider taking a short break.",
        "Rest is productive too.",
        "Step away from responsibilities briefly.",
        "Protect your mental energy."
        ],

        "fear":[
        "Pause and assess the situation calmly.",
        "Focus on facts rather than assumptions.",
        "Practice grounding techniques.",
        "Slow breathing may help."
        ],

        "joy":[
        "Keep spreading positivity.",
        "Celebrate this moment.",
        "Share your happiness with others.",
        "Express gratitude."
        ],

        "neutral":[
        "Your emotional state appears balanced.",
        "Maintain healthy routines.",
        "Stay mindful during the day.",
        "Reflect on your goals briefly."
        ]
    }

    return random.choice(advice_bank[primary])


def save_to_db(text,timestamp,scores,interpretation,suggestion):

    c.execute("""
    INSERT INTO history
    (text,timestamp,sadness,anxiety,emotional_fatigue,fear,joy,neutral,interpretation,suggestion)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """,(text,timestamp,
         scores["sadness"],
         scores["anxiety"],
         scores["emotional_fatigue"],
         scores["fear"],
         scores["joy"],
         scores["neutral"],
         interpretation,
         suggestion))

    conn.commit()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 GenSenti Navigation")
page = st.sidebar.radio("",["Home","GenSenti","History","Reports"])

# ---------------- HOME ----------------
if page=="Home":

    st.title("🧠 GenSenti")
    st.subheader("Gen-Z Emotion Intelligence System")

    st.write("GenSenti analyzes emotions in text using a transformer-based model enhanced with Gen-Z slang and emoji context.")

    st.write("The system provides emotional interpretation, supportive suggestions, and tracks emotional history for awareness.")

    st.markdown("""
    <div style='background-color:#330033;padding:20px;border-radius:10px;border:2px solid #ff00cc'>
    
    ⚠️ <b>Disclaimer</b>  
    GenSenti is designed for educational and awareness purposes only.  
    It does <b>not</b> provide medical or psychological diagnosis.

    </div>
    """, unsafe_allow_html=True)

# ---------------- ANALYZER ----------------
elif page=="GenSenti":

    st.title("💬 GenSenti Analyzer")

    text = st.text_area("Enter your text")

    if st.button("Analyze"):

        if text.strip():

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )

            with torch.inference_mode():

                logits = model(**inputs).logits
                probs = torch.sigmoid(logits)[0].tolist()

            scores = dict(zip(LABELS,probs))

            st.subheader("📊 Emotional Confidence Levels")

            for emotion,value in scores.items():

                st.write(f"{emotion.capitalize()} ({round(value*100,2)}%)")
                st.progress(float(value))

            interpretation = generate_explanation(text,scores)
            suggestion = generate_advice(scores)

            st.subheader("🧠 Interpretation")
            st.info(interpretation)

            st.subheader("💡 Suggestion")
            st.success(suggestion)

            save_to_db(text,timestamp,scores,interpretation,suggestion)

        else:
            st.warning("Please enter text.")

# ---------------- HISTORY ----------------
elif page=="History":

    st.title("📜 Emotional History")

    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC",conn)

    if not df.empty:
        st.dataframe(df,use_container_width=True)
    else:
        st.info("No history yet.")

# ---------------- REPORTS ----------------
elif page=="Reports":

    st.title("📥 Generate Report")

    df = pd.read_sql_query("SELECT * FROM history",conn)

    if not df.empty:

        st.dataframe(df,use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
        "Download CSV Report",
        csv,
        "gensenti_report.csv",
        "text/csv")

    else:
        st.info("No data available.")

# ---------------- DISCLAIMER ----------------
st.markdown("""
---
⚠️ **Disclaimer**  
GenSenti is intended for research and awareness purposes only.  
It does **not provide medical or psychological diagnosis**.
""")