import streamlit as st
import torch
import pandas as pd
import time
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="GenSenti", page_icon="🧠", layout="wide")

LABELS = ["sadness","anxiety","emotional_fatigue","fear","joy","neutral"]

MODEL_PATH = "gensenti_model"

# SESSION STORAGE
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# LOAD MODEL
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ADVICE SYSTEM
advice = {
"sadness":["Talk to someone you trust","Take time to rest","Go for a short walk"],
"anxiety":["Try slow breathing","Focus on what you control","Take a break"],
"emotional_fatigue":["Rest and recharge","Reduce tasks","Hydrate and relax"],
"fear":["Pause and assess calmly","Focus on facts","You can overcome this"],
"joy":["Share positivity","Celebrate this moment","Stay grateful"],
"neutral":["Your emotions look balanced","Maintain your routine","Stay mindful"]
}

# LOGIN PAGE
if not st.session_state.logged_in:

    st.title("🧠 GenSenti")
    st.subheader("AI Emotion Intelligence Platform")

    option = st.radio("Account",["Login","Register"])

    if option=="Register":

        user = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password",type="password")

        if st.button("Create Account"):

            if user in st.session_state.users:
                st.error("User already exists")

            else:
                st.session_state.users[user]={"email":email,"password":password}
                st.success("Account created successfully")

    if option=="Login":

        user = st.text_input("Username")
        password = st.text_input("Password",type="password")

        if st.button("Login"):

            if user in st.session_state.users and st.session_state.users[user]["password"]==password:
                st.session_state.logged_in=True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid login")

# MAIN APP
else:

    st.sidebar.title("GenSenti")

    page = st.sidebar.radio(
        "Navigation",
        ["Home","GenSenti Analyzer","History","Reports"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in=False
        st.rerun()

    if page=="Home":

        st.title("✨ GenSenti Dashboard")

        st.write(
        "GenSenti analyzes emotions in text including modern Gen-Z slang and expressions."
        )

        st.warning(
        "⚠️ Disclaimer: GenSenti is for educational awareness only."
        )

    if page=="GenSenti Analyzer":

        st.title("💬 Emotion Analyzer")

        text = st.text_area("Enter a sentence")

        if st.button("Analyze"):

            inputs = tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=64)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits)[0].tolist()

            scores = dict(zip(LABELS,probs))

            st.subheader("Emotion Confidence")

            for e,v in scores.items():
                st.write(f"{e} {round(v*100,2)}%")
                st.progress(v)

            emotion=max(scores,key=scores.get)

            explanation=f"The sentence mainly expresses **{emotion}**."

            suggestion=random.choice(advice[emotion])

            st.info(explanation)
            st.success(suggestion)

            st.session_state.history.append({
                "text":text,
                "emotion":emotion,
                "time":time.strftime("%Y-%m-%d %H:%M:%S")
            })

    if page=="History":

        st.title("📜 Analysis History")

        if len(st.session_state.history)==0:
            st.info("No analysis yet")

        else:
            df=pd.DataFrame(st.session_state.history)
            st.dataframe(df,use_container_width=True)

    if page=="Reports":

        st.title("📥 Download Report")

        if len(st.session_state.history)==0:
            st.info("No data available")

        else:

            df=pd.DataFrame(st.session_state.history)

            csv=df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV Report",
                csv,
                "gensenti_report.csv",
                "text/csv"
            )
