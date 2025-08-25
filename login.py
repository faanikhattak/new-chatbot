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
def login_page():
    st.title("🔐 Login to Chatbot")
    st.markdown("Please log in with your **Google Account** to continue.")

    # --- If user is already logged in ---
    if "user" in st.session_state:
        st.success(f"Welcome back, {st.session_state['user']['name']} 👋")
        if st.button("Logout"):
            st.session_state.pop("user", None)
            st.rerun()
        return

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
    query_params = st.experimental_get_query_params()
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
            st.experimental_rerun()
        else:
            st.error("Login failed. Please try again.")


