import streamlit as st
import sqlite3
import subprocess

# ---------------- DATABASE ----------------

conn = sqlite3.connect("src/gensenti.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
email TEXT,
password TEXT
)
""")

conn.commit()

# ---------------- UI ----------------

st.set_page_config(page_title="GenSenti Login", page_icon="🧠")

st.title("🧠 GenSenti")
st.subheader("Login / Register")

menu = st.radio("Select Option", ["Login", "Register"])

# ---------------- LOGIN ----------------

if menu == "Login":

    st.markdown("### Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()

        if user:
            st.success("Login Successful")

            st.info("Opening GenSenti Dashboard...")

            # Launch main app
            subprocess.Popen(["streamlit", "run", "src/app.py"])

        else:
            st.error("Invalid credentials")

# ---------------- REGISTER ----------------

if menu == "Register":

    st.markdown("### Create Account")

    new_user = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")

    if st.button("Register"):

        cursor.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (new_user, new_email, new_password)
        )

        conn.commit()

        st.success("Account Created Successfully")
        st.info("You can now login.")