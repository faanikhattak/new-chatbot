import streamlit as st
import requests
from urllib.parse import urlencode

# --- GOOGLE OAUTH CONFIG ---
CLIENT_ID = st.secrets["google_oauth"]["client_id"]
CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"

# Scopes
SCOPE = "openid email profile"
AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# --- LOGIN FUNCTION ---
import streamlit as st
from streamlit_oauth import OAuth2Component
import os

# Enable wide layout
st.set_page_config(page_title="Login Page", layout="centered")

def login_page():
    st.markdown("<h1 style='text-align: center; color: black;'>🔐 Login to Chatbot</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .login-card {
            background: white;
            color: black;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
            width: 400px;
            text-align: center;
            margin: auto;
        }
        .login-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        button {
            background-color: #2575fc !important;
            color: white !important;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    # st.markdown("<div class='login-title'>Welcome to Smart Chatbot</div>", unsafe_allow_html=True)
    # st.write("Sign in with your preferred method below:")

    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("🔑 Login with Google"):
    #         st.success("Redirecting to Google login... (Implement OAuth)")

    # with col2:
    #     if st.button("🐙 Login with GitHub"):
    #         st.success("Redirecting to GitHub login... (Implement OAuth)")

    # st.markdown("</div>", unsafe_allow_html=True)


    # --- Generate Google Auth URL ---
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent"
    }
    login_url = f"{AUTH_URL}?{urlencode(params)}"

    # --- Show login button ---
    st.markdown(f"[![Login with Google](https://img.shields.io/badge/Login%20with%20Google-blue?logo=google)]({login_url})")

    # --- Get Auth Code from URL ---
    query_params = st.query_params()
    if "code" in query_params:
        code = query_params["code"][0]

        # Exchange code for token
        token_data = {
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        token_response = requests.post(TOKEN_URL, data=token_data)
        token_json = token_response.json()

        if "access_token" in token_json:
            access_token = token_json["access_token"]

            # Get user info
            user_info_response = requests.get(
                USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_info = user_info_response.json()
            st.session_state["user"] = {
                "name": user_info.get("name", "User"),
                "email": user_info.get("email", "No email")
            }
            st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
            st.rerun()
        else:
            st.error("Login failed. Please try again.")





