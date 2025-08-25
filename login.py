import streamlit as st
import requests
from urllib.parse import urlencode

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Login Page", layout="wide")

# --- GOOGLE OAUTH CONFIG ---
CLIENT_ID = st.secrets["google_oauth"]["client_id"]
CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"  # اپنے ڈومین کے مطابق

SCOPE = "openid email profile"
AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def login_page():
    st.markdown(
        """
        <style>
        .login-container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            width: 450px;
            margin: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #2575fc;
        }
        .subtitle {
            font-size: 16px;
            color: #666;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>🔐 Login to Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Sign in to continue</div>", unsafe_allow_html=True)

    # Generate Google OAuth URL
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent"
    }
    login_url = f"{AUTH_URL}?{urlencode(params)}"

    st.markdown(f"[![Login with Google](https://img.shields.io/badge/Login%20with%20Google-blue?logo=google)]({login_url})")

    # Handle OAuth callback
    query_params = st.query_params
    if "code" in query_params:
        code = query_params["code"][0]
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
            user_info_response = requests.get(
                USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_info = user_info_response.json()
            st.session_state["is_authenticated"] = True
            st.session_state["user"] = {
                "name": user_info.get("name", "User"),
                "email": user_info.get("email", "No email")
            }
            st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
            st.rerun()
        else:
            st.error("Login failed. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)
