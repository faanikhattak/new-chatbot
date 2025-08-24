import streamlit as st
import requests
import urllib.parse

# Initialize session state
if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}

# --- OAUTH AUTHENTICATION FUNCTIONS ---
def get_auth_url(provider, client_id, redirect_uri):
    if provider == "google":
        return (
            "https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={client_id}&response_type=code&"
            f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
            "scope=openid%20email%20profile"
        )
    elif provider == "github":
        return (
            "https://github.com/login/oauth/authorize?"
            f"client_id={client_id}&redirect_uri={urllib.parse.quote(redirect_uri)}"
        )
    else:
        raise ValueError("Unsupported provider")

def exchange_code_for_token(provider, code, client_id, client_secret, redirect_uri):
    if provider == "google":
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    elif provider == "github":
        token_url = "https://github.com/login/oauth/access_token"
        payload = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
    else:
        raise ValueError("Unsupported provider")

    headers = {"Accept": "application/json"}
    try:
        response = requests.post(token_url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error exchanging code for token: {e}")
        return None

def get_user_info(provider, token_data):
    if provider == "google":
        user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    elif provider == "github":
        user_info_url = "https://api.github.com/user"
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    else:
        raise ValueError("Unsupported provider")

    try:
        response = requests.get(user_info_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching user info: {e}")
        return None

def handle_oauth_callback(provider, client_id, client_secret, redirect_uri):
    """Handles OAuth callback using new st.query_params"""
    params = st.query_params
    code_list = params.get("code")
    if code_list:
        code = code_list[0]
        token_data = exchange_code_for_token(provider, code, client_id, client_secret, redirect_uri)
        if token_data:
            user_info = get_user_info(provider, token_data)
            if user_info:
                st.session_state["user_info"] = user_info
                st.session_state["is_authenticated"] = True
                # Clear query params after successful login
                st.experimental_set_query_params()  # This can be replaced with st.session_state["query_params"] = {} if needed

# --- DISPLAY LOGIN PAGE ---
def login_page():
    st.title("🔐 Login to AI Chatbot")
    st.write("Please login using Google or GitHub")

    redirect_uri = "http://localhost:8501"  # Change if deployed
    google_client_id = st.secrets["google_oauth"]["client_id"]
    google_client_secret = st.secrets["google_oauth"]["client_secret"]
    github_client_id = st.secrets["github_oauth"]["client_id"]
    github_client_secret = st.secrets["github_oauth"]["client_secret"]

    if st.button("Login with Google"):
        auth_url = get_auth_url("google", google_client_id, redirect_uri)
        st.markdown(f"[Click here to continue]({auth_url})")

    if st.button("Login with GitHub"):
        auth_url = get_auth_url("github", github_client_id, redirect_uri)
        st.markdown(f"[Click here to continue]({auth_url})")

    # Handle OAuth callback automatically
    if "code" in st.query_params:
        handle_oauth_callback("google", google_client_id, google_client_secret, redirect_uri)
