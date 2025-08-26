import streamlit as st
import requests
from urllib.parse import urlencode, parse_qsl
from authlib.integrations.requests_client import OAuth2Session
import json
import streamlit as st
from urllib.parse import urlencode
import requests
# --- PAGE SETTINGS ---
st.set_page_config(page_title="Login Page", layout="centered")

# --- OAUTH CONFIG (Google + GitHub) ---
# Note: In a real app, these should be securely stored in st.secrets
# GOOGLE_CLIENT_ID = st.secrets.get("google_oauth", {}).get("client_id")
# GOOGLE_CLIENT_SECRET = st.secrets.get("google_oauth", {}).get("client_secret")

# GITHUB_CLIENT_ID = st.secrets.get("github_oauth", {}).get("client_id")
# GITHUB_CLIENT_SECRET = st.secrets.get("github_oauth", {}).get("client_secret")

# # In a deployed app, make sure this is your app's public URL
# REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"

#[google_oauth]
GOOGLE_CLIENT_ID= "139553575448-buecvcrqieq89l3jhvhrq7a8o0lkt26e.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET= "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"


#[github_oauth]
GITHUB_CLIENT_ID = "Ov23liYmL2IV9XYzi19E"
GITHUB_CLIENT_SECRET= "1a115e96a5ff9dc0735abe76f9af7d836fa7a29e"
REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"



#[api_keys]
HF_TOKEN = "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"

# --- Endpoints ---
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"


def login_page():
    # --- CSS Styling ---
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
        }
        .login-card {
            background: white;
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
        .google-btn { background-color: #DB4437; }
        .github-btn { background-color: #333; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>🔐 Login to Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-subtitle'>Sign in with your preferred account</div>", unsafe_allow_html=True)

    # --- Generate Google URL with state ---
    google_params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": "google"  # Add a state parameter to identify the provider
    }
    google_login_url = f"{GOOGLE_AUTH_URL}?{urlencode(google_params)}"

    # --- Generate GitHub URL with state ---
    github_params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "read:user user:email",
        "state": "github" # Add a state parameter to identify the provider
    }
    github_login_url = f"{GITHUB_AUTH_URL}?{urlencode(github_params)}"

    # --- Show Buttons ---
    st.markdown(f"<a href='{google_login_url}' class='login-btn google-btn'>Login with Google</a>", unsafe_allow_html=True)
    st.markdown(f"<a href='{github_login_url}' class='login-btn github-btn'>Login with GitHub</a>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Handle Callback ---
    query_params = st.query_params
    if "code" in query_params:
        code = query_params["code"][0]
        provider = query_params.get("state")

        if provider == "google":
            # Handle Google login
            try:
                oauth = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
                                      redirect_uri=REDIRECT_URI)
                token_response = oauth.fetch_token(
                    GOOGLE_TOKEN_URL,
                    code=code,
                    grant_type="authorization_code"
                )
                access_token = token_response.get("access_token")
                user_info = requests.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                ).json()
                st.session_state["is_authenticated"] = True
                st.session_state["user"] = {"name": user_info.get("name"), "email": user_info.get("email")}
                st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
                # Clear query parameters to prevent re-using the expired code
                st.experimental_set_query_params(code=None, state=None)
                st.rerun()
            except Exception as e:
                st.error(f"Google login failed. Error: {e}")
        elif provider == "github":
            # Handle GitHub login
            try:
                # Use authlib to properly handle the GitHub token exchange
                oauth = OAuth2Session(GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
                                      redirect_uri=REDIRECT_URI)
                token_response = oauth.fetch_token(
                    GITHUB_TOKEN_URL,
                    code=code,
                    headers={"Accept": "application/json"}
                )
                access_token = token_response.get("access_token")

                if access_token:
                    user_info = requests.get(
                        GITHUB_USERINFO_URL,
                        headers={"Authorization": f"Bearer {access_token}"}
                    ).json()
                    st.session_state["is_authenticated"] = True
                    st.session_state["user"] = {"name": user_info.get("login"), "email": user_info.get("email")}
                    st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
                    # Clear query parameters to prevent re-using the expired code
                    st.experimental_set_query_params(code=None, state=None)
                    st.rerun()
                else:
                    st.error("GitHub login failed. No access token received.")
            except Exception as e:
                st.error(f"GitHub login failed. Error: {e}")


# Main application logic
if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False

if not st.session_state["is_authenticated"]:
    login_page()
else:
    st.title("Logged In!")
    st.write(f"Hello, {st.session_state['user'].get('name')}!")
    st.write(f"Your email is: {st.session_state['user'].get('email')}")
    if st.button("Logout"):
        st.session_state["is_authenticated"] = False
        if "user" in st.session_state:
            del st.session_state["user"]
        st.experimental_set_query_params(code=None, state=None)
        st.rerun()






# import streamlit as st
# import requests
# from urllib.parse import urlencode

# # --- PAGE SETTINGS ---
# st.set_page_config(page_title="Login Page", layout="centered")

# # --- OAUTH CONFIG (Google + GitHub) ---
# GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["client_id"]
# GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]

# GITHUB_CLIENT_ID = st.secrets["github_oauth"]["client_id"]
# GITHUB_CLIENT_SECRET = st.secrets["github_oauth"]["client_secret"]

# REDIRECT_URI = "https://new-chatbot-faani.streamlit.app/"  # Change to your domain

# # --- Endpoints ---
# GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
# GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
# GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
# GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
# GITHUB_USERINFO_URL = "https://api.github.com/user"


# def login_page():
    
#     # --- CSS Styling ---
#     st.markdown(
#         """
#         <style>
#         body {
#             background: linear-gradient(to right, #6a11cb, #2575fc);
#         }
#         .login-card {
#             background: black;
#             padding: 40px;
#             border-radius: 20px;
#             width: 450px;
#             text-align: center;
#             margin: 80px auto;
#             box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
#         }
#         .login-title {
#             font-size: 28px;
#             font-weight: bold;
#             color: #2575fc;
#             margin-bottom: 10px;
#         }
#         .login-subtitle {
#             font-size: 16px;
#             color: #666;
#             margin-bottom: 30px;
#         }
#         .login-btn {
#             display: inline-block;
#             width: 80%;
#             padding: 12px;
#             margin: 10px 0;
#             font-size: 18px;
#             border-radius: 10px;
#             text-decoration: none;
#             color: white !important;
#             font-weight: bold;
#         }
#         .google-btn {
#             background-color: #DB4437;
#         }
#         .github-btn {
#             background-color: #333;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

    
#     st.markdown("<div class='login-title'>🔐 Login to Chatbot</div>", unsafe_allow_html=True)
#     st.markdown("<div class='login-subtitle'>Sign in with your preferred account</div>", unsafe_allow_html=True)

#     # --- Google Login URL ---
#     google_params = {
#         "client_id": GOOGLE_CLIENT_ID,
#         "redirect_uri": REDIRECT_URI,
#         "response_type": "code",
#         "scope": "openid email profile",
#         "access_type": "offline",
#         "prompt": "consent"
#     }
#     google_login_url = f"{GOOGLE_AUTH_URL}?{urlencode(google_params)}"

#     # --- GitHub Login URL ---
#     github_params = {
#         "client_id": GITHUB_CLIENT_ID,
#         "redirect_uri": REDIRECT_URI,
#         "scope": "read:user user:email"
#     }
#     github_login_url = f"{GITHUB_AUTH_URL}?{urlencode(github_params)}"

#     # --- Buttons ---
#     st.markdown(f"<a class='login-btn google-btn' href='{google_login_url}'>🌐 Login with Google</a>", unsafe_allow_html=True)
#     st.markdown(f"<a class='login-btn github-btn' href='{github_login_url}'>🐙 Login with GitHub</a>", unsafe_allow_html=True)

#     # --- Handle Callback ---
#     query_params = st.query_params
#     if "code" in query_params:
#         code = query_params["code"][0]

#         # --- Google OAuth ---
#         if "google" in str(query_params):
#             token_data = {
#                 "code": code,
#                 "client_id": GOOGLE_CLIENT_ID,
#                 "client_secret": GOOGLE_CLIENT_SECRET,
#                 "redirect_uri": REDIRECT_URI,
#                 "grant_type": "authorization_code"
#             }
#             token_response = requests.post(GOOGLE_TOKEN_URL, data=token_data)
#             token_json = token_response.json()

#             if "access_token" in token_json:
#                 access_token = token_json["access_token"]
#                 user_info = requests.get(
#                     GOOGLE_USERINFO_URL,
#                     headers={"Authorization": f"Bearer {access_token}"}
#                 ).json()
#                 st.session_state["is_authenticated"] = True
#                 st.session_state["user"] = {"name": user_info.get("name"), "email": user_info.get("email")}
#                 st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
#                 st.rerun()
#             else:
#                 st.error("Google login failed. Try again.")

#         # --- GitHub OAuth ---
#         else:
#             token_data = {
#                 "client_id": GITHUB_CLIENT_ID,
#                 "client_secret": GITHUB_CLIENT_SECRET,
#                 "code": code,
#                 "redirect_uri": REDIRECT_URI
#             }
#             token_response = requests.post(GITHUB_TOKEN_URL, data=token_data, headers={"Accept": "application/json"})
#             token_json = token_response.json()

#             if "access_token" in token_json:
#                 access_token = token_json["access_token"]
#                 user_info = requests.get(
#                     GITHUB_USERINFO_URL,
#                     headers={"Authorization": f"Bearer {access_token}"}
#                 ).json()
#                 st.session_state["is_authenticated"] = True
#                 st.session_state["user"] = {"name": user_info.get("login"), "email": user_info.get("email")}
#                 st.success(f"Welcome, {st.session_state['user']['name']} 🎉")
#                 st.rerun()
#             else:
#                 st.error("GitHub login failed. Try again.")

#     st.markdown("</div>", unsafe_allow_html=True)











