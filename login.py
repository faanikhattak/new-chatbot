import streamlit as st
import requests
from urllib.parse import urlencode

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Login Page", layout="centered")

# --- OAUTH CONFIG (Google + GitHub) ---
GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]

GITHUB_CLIENT_ID = st.secrets["github_oauth"]["client_id"]
GITHUB_CLIENT_SECRET = st.secrets["github_oauth"]["client_secret"]

REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"  # Change to your domain

# --- Endpoints ---
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"


def login_page():
    st.markdown("<h1 style='text-align: center; color: black;'>🔐 Login to Chatbot</h1>", unsafe_allow_html=True)
    # --- CSS Styling ---
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
        }
        .login-card {
            background: black;
            padding: 40px;
            border-radius: 20px;
            width: 450px;
            text-align: center;
            margin: 80px auto;
            box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
        }
        .login-title {
            font-size: 28px;
            font-weight: bold;
            color: #2575fc;
            margin-bottom: 10px;
        }
        .login-subtitle {
            font-size: 16px;
            color: #666;
            margin-bottom: 30px;
        }
        .login-btn {
            display: inline-block;
            width: 80%;
            padding: 12px;
            margin: 10px 0;
            font-size: 18px;
            border-radius: 10px;
            text-decoration: none;
            color: white !important;
            font-weight: bold;
        }
        .google-btn {
            background-color: #DB4437;
        }
        .github-btn {
            background-color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown("<div class='login-title'>🔐 Login to Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-subtitle'>Sign in with your preferred account</div>", unsafe_allow_html=True)

    # --- Google Login URL ---
    google_params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    google_login_url = f"{GOOGLE_AUTH_URL}?{urlencode(google_params)}"

    # --- GitHub Login URL ---
    github_params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "read:user user:email"
    }
    github_login_url = f"{GITHUB_AUTH_URL}?{urlencode(github_params)}"

    # --- Buttons ---
    st.markdown(f"<a class='login-btn google-btn' href='{google_login_url}'>🌐 Login with Google</a>", unsafe_allow_html=True)
    st.markdown(f"<a class='login-btn github-btn' href='{github_login_url}'>🐙 Login with GitHub</a>", unsafe_allow_html=True)

    # --- Handle Callback ---
    query_params = st.query_params
    if "code" in query_params:
        code = query_params["code"][0]

        # --- Google OAuth ---
        if "google" in str(query_params):
            token_data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            token_response = requests.post(GOOGLE_TOKEN_URL, data=token_data)
            token_json = token_response.json()

            if "access_token" in token_json:
                access_token = token_json["access_token"]
                user_info = requests.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                ).json()
                st.session_state["is_authenticated"] = True
                st.session_state["user"] = {"name": user_info.get("name"), "email": user_info.get("email")}
                st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
                st.rerun()
            else:
                st.error("Google login failed. Try again.")

        # --- GitHub OAuth ---
        else:
            token_data = {
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": REDIRECT_URI
            }
            token_response = requests.post(GITHUB_TOKEN_URL, data=token_data, headers={"Accept": "application/json"})
            token_json = token_response.json()

            if "access_token" in token_json:
                access_token = token_json["access_token"]
                user_info = requests.get(
                    GITHUB_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                ).json()
                st.session_state["is_authenticated"] = True
                st.session_state["user"] = {"name": user_info.get("login"), "email": user_info.get("email")}
                st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
                st.rerun()
            else:
                st.error("GitHub login failed. Try again.")

    st.markdown("</div>", unsafe_allow_html=True)




