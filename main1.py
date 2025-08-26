# # #
# # # This Streamlit app creates an AI chatbot for your documents.
# # # It allows you to upload files, indexes them, and then
# # # lets you ask questions about them.
# # #
# # # Dependencies:
# # # pip install streamlit langchain pandas pypdf unstructured openai tiktoken
# # # pip install faiss-cpu sentence-transformers torch huggingface_hub
# # #

# import os
# import shutil
# import json
# import io
# from datetime import datetime
# import torch
# import pandas as pd
# import streamlit as st
# from time import sleep

# # LangChain Imports
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.llms.base import LLM
# from langchain.llms.utils import enforce_stop_tokens
# from langchain.callbacks.manager import CallbackManagerForLLMRun # <-- Corrected: This import was missing!
# from langchain_community.document_loaders import (
#     PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
# )
# from langchain_openai import ChatOpenAI
# from openai import OpenAI as OpenAIApiClient

# # Hugging Face Imports for local model
# from huggingface_hub import InferenceClient
# from typing import Any, List, Optional, Dict
# from pydantic import Extra

# # --- CONFIGURATION ---
# INDEX_PATH = "faiss_index"
# TEMP_DIR = "temp_documents"
# st.set_page_config(page_title="AI Chatbot for Your Documents", layout="wide")
# st.title("AI Chatbot for Your Documents 🤖")

# # --- NEW CUSTOM LLM CLASS ---
# # This class acts as a bridge between the HuggingFace InferenceClient and LangChain's LLM interface.
# # It allows us to use the modern InferenceClient within the LangChain framework.
# class CustomHuggingFaceLLM(LLM):
#     """A custom LangChain LLM wrapper for the HuggingFace InferenceClient."""
    
#     client: InferenceClient
#     model: str = "deepseek-ai/DeepSeek-V3-0324"
#     max_tokens: int = 512
#     temperature: float = 0.3
    
#     class Config:
#         extra = "allow"

#     @property
#     def _llm_type(self) -> str:
#         return "huggingface_inference_client"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """The main method to call the LLM."""
        
#         # Use the chat completions endpoint for better performance
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=self.max_tokens,
#             temperature=self.temperature
#         )
        
#         # Extract the text from the response
#         text = response.choices[0].message.content

#         # Handle stop tokens, if any
#         if stop is not None:
#             text = enforce_stop_tokens(text, stop)
        
#         return text

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         """Get the identifying parameters."""
#         return {
#             "model": self.model,
#             "max_tokens": self.max_tokens,
#             "temperature": self.temperature
#         }

# # --- SIDEBAR WIDGETS ---
# def sidebar_widgets():
#     """Sidebar mein UI elements banata hai jaise ki model selection,
#     API key input, aur file uploader.
#     """
#     st.sidebar.title("⚙️ Settings")
#     enable_smart_mode = st.sidebar.checkbox("Enable Smart Thinking Mode ", value=False, key="smart_mode_checkbox")

#     model_choice = "Local Model"
#     openai_key = ""
#     api_key_valid = True  # Default to True for local model
#     openai_models = ["gpt-3.5-turbo"]

#     if enable_smart_mode:
#         openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
#         st.sidebar.subheader("Model Selection")
#         model_choice = st.sidebar.selectbox(
#             "Choose LLM Model",
#             ["Local Model", "OpenAI (Cloud)"],
#             key="llm_model_type_selectbox"
#         )

#     if model_choice == "OpenAI (Cloud)":
#         st.sidebar.subheader("OpenAI Model")
#         openai_model_selected = st.sidebar.selectbox("Choose OpenAI Model", openai_models, key="openai_model_selectbox")
#         openai_key = st.sidebar.text_input("Enter OpenAI API Key 🔑", type="password", key="openai_api_key_input")
#         submit_key_button = st.sidebar.button("Submit API Key", key="submit_api_key_button")

#         if submit_key_button and openai_key:
#             st.session_state["openai_api_key"] = openai_key
#             try:
#                 # Key ko validate karne ke liye ek chota sa test call karen
#                 client = OpenAIApiClient(api_key=openai_key)
#                 client.models.list()
#                 st.sidebar.success("✅ API Key is valid!")
#                 api_key_valid = True
#                 st.session_state["api_key_submitted"] = True
#                 st.session_state["openai_model"] = openai_model_selected
#             except Exception as e:
#                 st.sidebar.error(f"❌ Invalid API Key: {e}")
#                 st.session_state["api_key_submitted"] = False
#                 api_key_valid = False
#                 st.session_state["openai_model"] = None
#         else:
#             st.session_state["openai_model"] = openai_model_selected
#     else:
#         st.session_state["openai_api_key"] = ""
#         st.session_state["api_key_submitted"] = False
#         st.session_state["openai_model"] = None
#         api_key_valid = True

#     st.sidebar.title("📁 Document Input")
#     files = st.sidebar.file_uploader("📄 Upload Documents", type=["pdf", "txt", "md", "docx", "csv"], accept_multiple_files=True, key="file_uploader")
#     process = st.sidebar.button("🔄 Process & Index Files", key="process_button")
#     return model_choice, files, process, api_key_valid

# # --- EMBEDDING MODEL ---
# @st.cache_resource
# def load_embedding_model():
#     """HuggingFace embeddings model ko load karta hai aur cache karta hai."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# embedding_model = load_embedding_model()

# # --- SANITIZATION & UTILITIES ---
# def remove_surrogates(text):
#     """Text se surrogate characters hataata hai."""
#     try:
#         return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore') if isinstance(text, str) else text
#     except:
#         return ""

# def clean_chat_history(chat_history):
#     """Chat history se surrogate characters ko saaf karta hai."""
#     return [
#         {
#             "role": remove_surrogates(msg["role"]),
#             "content": remove_surrogates(msg["content"]),
#             "timestamp": remove_surrogates(msg.get("timestamp", str(datetime.now())))
#         }
#         for msg in chat_history
#     ]

# # --- DOCUMENT LOADING ---
# def load_documents(files):
#     """Uploaded files ko process karta hai aur LangChain Document objects mein load karta hai."""
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     docs = []
#     for file in files:
#         path = os.path.join(TEMP_DIR, file.name)
#         with open(path, "wb") as f:
#             f.write(file.getbuffer())
#         ext = os.path.splitext(file.name)[-1].lower()
#         try:
#             if ext == ".pdf":
#                 docs.extend(PyPDFLoader(path).load())
#             elif ext == ".txt":
#                 docs.extend(TextLoader(path).load())
#             elif ext == ".md":
#                 docs.extend(UnstructuredMarkdownLoader(path).load())
#             elif ext == ".docx":
#                 docs.extend(UnstructuredWordDocumentLoader(path).load())
#             elif ext == ".csv":
#                 df = pd.read_csv(path, sep=None, engine="python")
#                 content = df.to_string(index=False)
#                 docs.append(Document(page_content=content, metadata={"source": file.name}))
#             else:
#                 st.warning(f"⚠️ Unsupported file: {file.name}")
#         except Exception as e:
#                 st.error(f"❌ Error loading {file.name}: {e}")
#     return docs

# # --- VECTOR STORE ---
# def build_vector_store(docs):
#     """Documents se ya existing se ek FAISS vector store banata ya update karta hai."""
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     split_docs = splitter.split_documents(docs)
#     try:
#         if os.path.exists(INDEX_PATH):
#             store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
#             store.add_documents(split_docs)
#         else:
#             store = FAISS.from_documents(split_docs, embedding_model)
#         store.save_local(INDEX_PATH)
#         return store
#     except Exception as e:
#         st.warning(f"⚠️ Indexing issue: {e}")
#         return None

# # --- LLM SELECTION ---
# def get_llm():
#     """Initializes and returns the appropriate LLM instance."""
    
#     # Check if a valid OpenAI key has been submitted
#     if st.session_state.get("api_key_submitted") and st.session_state.get("openai_api_key"):
#         openai_model = st.session_state.get("openai_model", "gpt-3.5-turbo")
#         st.sidebar.write(f"✅ Using OpenAI: {openai_model}")
#         return ChatOpenAI(
#             temperature=0.3,
#             openai_api_key=st.session_state['openai_api_key'],
#             model=openai_model
#         )
    
#     else:
#         # Check for Hugging Face token from environment variables
        
#         hf_token = os.getenv("HF_TOKEN")
#         # I've set a temporary default token based on the user's "working" snippet for demonstration purposes.
#         # In a real app, it's best to handle this securely, e.g., using Streamlit secrets.
#         if not hf_token:
#             st.sidebar.warning("⚠️ HF_TOKEN environment variable not set. Using a hardcoded token for this example.")

#         if not hf_token:
#             st.sidebar.error("❌ Error: HF_TOKEN is not available. Local model cannot be initialized.")
#             return None

#         # Use the custom LLM class with Hugging Face InferenceClient
#         st.sidebar.write("✅ Using DeepSeek-V3 (HuggingFace API)")
#         client = InferenceClient(token=hf_token)
        
#         # Instantiate the custom LLM
#         return CustomHuggingFaceLLM(
#             client=client,
#             model="deepseek-ai/DeepSeek-V3-0324",
#             max_tokens=512,
#             temperature=0.3
#         )

# # --- CHATBOT INITIALIZATION ---
# def initialize_chatbot():
#     """Chatbot chain ko shuru karta hai agar FAISS index maujood hai."""
#     if not os.path.exists(INDEX_PATH):
#         return None
#     try:
#         store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
#         # LLM instance ko get_llm function se prapt karen
#         llm = get_llm()
#         if llm:
#             return ConversationalRetrievalChain.from_llm(llm=llm, retriever=store.as_retriever(), memory=memory)
#         else:
#             return None
#     except Exception as e:
#         st.error(f"❌ Chatbot init failed: {e}")
#         return None

# # --- MAIN APP LOGIC ---
# # Sidebar widgets ko load karen aur session state ko shuru karen
# model_choice, uploaded_files, process_button, api_key_valid = sidebar_widgets()
# st.session_state.setdefault("chat_history", [])
# st.session_state.setdefault("documents", [])
# st.session_state.setdefault("model_choice", "Local Model") # Add to session state
# st.session_state.setdefault("qa_chain", None)
# st.session_state.setdefault("openai_model", None)

# # Agar model choice badal gai hai, to session state update karen
# if "last_model_choice" not in st.session_state or st.session_state["last_model_choice"] != model_choice:
#     st.session_state["model_choice"] = model_choice
#     st.session_state["last_model_choice"] = model_choice
#     st.session_state["qa_chain"] = initialize_chatbot() # Re-initialize the chain
#     st.rerun()

# # Document processing
# if uploaded_files and process_button:
#     with st.spinner("🔍 Indexing documents..."):
#         docs = load_documents(uploaded_files)
#         if docs:
#             st.session_state["documents"].extend(docs)
#             build_vector_store(st.session_state["documents"])
#             shutil.rmtree(TEMP_DIR)
#             st.success("✅ Upload Success! Documents indexed and ready for chat.")
#             st.session_state.qa_chain = initialize_chatbot()
#         else:
#             st.warning("⚠️ No valid documents found to process.")

# # Download & clear memory
# if st.session_state.chat_history:
#     try:
#         history_json = json.dumps(clean_chat_history(st.session_state.chat_history), indent=2, ensure_ascii=False)
#         chat_bytes = io.BytesIO(history_json.encode("utf-8"))
#         st.sidebar.download_button("📅 Download Chat History", chat_bytes, "chat_history.json", "application/json")
#     except Exception as e:
#         st.sidebar.error(f"❌ Download failed: {e}")

#     if st.sidebar.button("😹 Clear Memory"):
#         if os.path.exists(INDEX_PATH):
#             shutil.rmtree(INDEX_PATH)
#         st.session_state["documents"] = []
#         st.session_state["chat_history"] = []
#         st.session_state["qa_chain"] = None
#         st.success("✅ Memory and chat history cleared!")

# # --- CUSTOM CSS ---
# st.markdown("""
#     <style>
#     /* Spinner customization */
#     .stSpinner > div > div {
#         border-top-color: #FF69B4; /* Pink */
#         border-right-color: #1E90FF; /* Blue */
#         border-bottom-color: #32CD32; /* Green */
#         border-left-color: #FFD700; /* Gold */
#     }

#     /* Chat bubbles */
#     .user-bubble {
#         background-color: #DCF8C6;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         max-width: 80%;
#         align-self: flex-end;
#         text-align: left;
#     }
#     .bot-bubble {
#         background-color: #E6E6FA;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         max-width: 80%;
#         align-self: flex-start;
#         text-align: left;
#     }

#     /* Colorful words */
#     .thinking-word {
#         color: #FF69B4;
#         font-weight: bold;
#         font-size: 20px;
#     }
#     .query-word {
#         color: #1E90FF;
#         font-weight: bold;
#         font-size: 20px;
#     }
#     .blinking-cursor {
#         animation: blink 1s infinite;
#     }
#     @keyframes blink {
#         0% { opacity: 1; }
#         50% { opacity: 0; }
#         100% { opacity: 1; }
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- CHAT DISPLAY & INPUT ---
# if st.session_state.qa_chain:
#     chat_placeholder = st.container()

#     with chat_placeholder:
#         for msg in st.session_state.chat_history:
#             if msg["role"] == "user":
#                 st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

#     prompt = st.chat_input("Ask your questions here...")

#     if prompt:
#         # User ka message chat history mein jodein
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         st.markdown(
#             f"<span class='thinking-word'>You asked:</span> "
#             f"<span class='query-word'>{prompt}</span>",
#             unsafe_allow_html=True
#         )

#         with st.spinner('Generating...'):
#             try:
#                 # QA chain chalaayein
#                 response = st.session_state.qa_chain.invoke({"question": prompt})
#                 bot_reply = response["answer"] if "answer" in response else "No relevant information found."

#                 with st.chat_message(""):
#                     word_container = st.empty()
#                     full_response = ""
#                     for word in bot_reply.split():
#                         full_response += word + " "
#                         word_container.markdown(f'<div class="bot-bubble">{full_response}<span class="blinking-cursor">|</span></div>', unsafe_allow_html=True)
#                         sleep(0.05)
#                     st.session_state.chat_history.append({"role": "bot", "content": full_response})

#             except Exception as e:
#                 error_message = f"❌ Error: {e}"
#                 st.session_state.chat_history.append({"role": "bot", "content": error_message})

#         st.rerun()

# else:
#     st.info("📄 Please upload and process documents to begin chatting.")

















# #
# # This Streamlit app creates an AI chatbot for your documents.
# # It allows you to upload files, indexes them, and then
# # lets you ask questions about them.
# #
# # Dependencies:
# # pip install streamlit langchain pandas pypdf unstructured openai tiktoken
# # pip install faiss-cpu sentence-transformers torch huggingface_hub
# #

# import os
# import shutil
# import json
# import io
# from datetime import datetime
# import torch
# import pandas as pd
# import streamlit as st
# from time import sleep

# # LangChain Imports
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.llms.base import LLM
# from langchain.llms.utils import enforce_stop_tokens
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain_community.document_loaders import (
#     PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
# )
# from langchain.prompts import PromptTemplate  # <-- Added: Import PromptTemplate
# from langchain_openai import ChatOpenAI
# from openai import OpenAI as OpenAIApiClient

# # Hugging Face Imports for local model
# from huggingface_hub import InferenceClient
# from typing import Any, List, Optional, Dict
# from pydantic import Extra

# # --- CONFIGURATION ---
# INDEX_PATH = "faiss_index"
# TEMP_DIR = "temp_documents"
# st.set_page_config(page_title="AI Chatbot for Your Documents", layout="wide")
# st.title("AI Chatbot for Your Documents 🤖")

# # --- NEW CUSTOM LLM CLASS ---
# # This class acts as a bridge between the HuggingFace InferenceClient and LangChain's LLM interface.
# # It allows us to use the modern InferenceClient within the LangChain framework.
# class CustomHuggingFaceLLM(LLM):
#     """A custom LangChain LLM wrapper for the HuggingFace InferenceClient."""
    
#     client: InferenceClient
#     model: str = "deepseek-ai/DeepSeek-V3-0324"
#     max_tokens: int = 512
#     temperature: float = 0.3
    
#     class Config:
#         extra = "allow"

#     @property
#     def _llm_type(self) -> str:
#         return "huggingface_inference_client"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """The main method to call the LLM."""
        
#         # Use the chat completions endpoint for better performance
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=self.max_tokens,
#             temperature=self.temperature
#         )
        
#         # Extract the text from the response
#         text = response.choices[0].message.content

#         # Handle stop tokens, if any
#         if stop is not None:
#             text = enforce_stop_tokens(text, stop)
        
#         return text

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         """Get the identifying parameters."""
#         return {
#             "model": self.model,
#             "max_tokens": self.max_tokens,
#             "temperature": self.temperature
#         }

# # --- SIDEBAR WIDGETS ---
# def sidebar_widgets():
#     """Sidebar mein UI elements banata hai jaise ki model selection,
#     API key input, aur file uploader.
#     """
#     st.sidebar.title("⚙️ Settings")
#     enable_smart_mode = st.sidebar.checkbox("Enable Smart Thinking Mode ", value=False, key="smart_mode_checkbox")

#     model_choice = "Open Sorce Model"
#     openai_key = ""
#     api_key_valid = True  # Default to True for local model
#     openai_models = ["gpt-3.5-turbo"]

#     if enable_smart_mode:
#         openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
#         st.sidebar.subheader("Model Selection")
#         model_choice = st.sidebar.selectbox(
#             "Choose LLM Model",
#             ["Open Sorce Model", "OpenAI better response (paid)"],
#             key="llm_model_type_selectbox"
#         )

#     if model_choice == "OpenAI better response (paid)":
#         st.sidebar.subheader("OpenAI better response (paid)")
#         openai_model_selected = st.sidebar.selectbox("Choose OpenAI Model", openai_models, key="openai_model_selectbox")
#         openai_key = st.sidebar.text_input("Enter OpenAI API Key 🔑", type="password", key="openai_api_key_input")
#         submit_key_button = st.sidebar.button("Submit API Key", key="submit_api_key_button")

#         if submit_key_button and openai_key:
#             st.session_state["openai_api_key"] = openai_key
#             try:
#                 # Key ko validate karne ke liye ek chota sa test call karen
#                 client = OpenAIApiClient(api_key=openai_key)
#                 client.models.list()
#                 st.sidebar.success("✅ API Key is valid!")
#                 api_key_valid = True
#                 st.session_state["api_key_submitted"] = True
#                 st.session_state["openai_model"] = openai_model_selected
#             except Exception as e:
#                 st.sidebar.error(f"❌ Invalid API Key: {e}")
#                 st.session_state["api_key_submitted"] = False
#                 api_key_valid = False
#                 st.session_state["openai_model"] = None
#         else:
#             st.session_state["openai_model"] = openai_model_selected
#     else:
#         st.session_state["openai_api_key"] = ""
#         st.session_state["api_key_submitted"] = False
#         st.session_state["openai_model"] = None
#         api_key_valid = True

#     st.sidebar.title("📁 Document Input")
#     files = st.sidebar.file_uploader("📄 Upload Documents", type=["pdf", "txt", "md", "docx", "csv"], accept_multiple_files=True, key="file_uploader")
#     process = st.sidebar.button("🔄 Process & Index Files", key="process_button")
#     return model_choice, files, process, api_key_valid

# # --- EMBEDDING MODEL ---
# @st.cache_resource
# def load_embedding_model():
#     """HuggingFace embeddings model ko load karta hai aur cache karta hai."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# embedding_model = load_embedding_model()

# # --- SANITIZATION & UTILITIES ---
# def remove_surrogates(text):
#     """Text se surrogate characters hataata hai."""
#     try:
#         return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore') if isinstance(text, str) else text
#     except:
#         return ""

# def clean_chat_history(chat_history):
#     """Chat history se surrogate characters ko saaf karta hai."""
#     return [
#         {
#             "role": remove_surrogates(msg["role"]),
#             "content": remove_surrogates(msg["content"]),
#             "timestamp": remove_surrogates(msg.get("timestamp", str(datetime.now())))
#         }
#         for msg in chat_history
#     ]

# # --- DOCUMENT LOADING ---
# def load_documents(files):
#     """Uploaded files ko process karta hai aur LangChain Document objects mein load karta hai."""
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     docs = []
#     for file in files:
#         path = os.path.join(TEMP_DIR, file.name)
#         with open(path, "wb") as f:
#             f.write(file.getbuffer())
#         ext = os.path.splitext(file.name)[-1].lower()
#         try:
#             if ext == ".pdf":
#                 docs.extend(PyPDFLoader(path).load())
#             elif ext == ".txt":
#                 docs.extend(TextLoader(path).load())
#             elif ext == ".md":
#                 docs.extend(UnstructuredMarkdownLoader(path).load())
#             elif ext == ".docx":
#                 docs.extend(UnstructuredWordDocumentLoader(path).load())
#             elif ext == ".csv":
#                 df = pd.read_csv(path, sep=None, engine="python")
#                 content = df.to_string(index=False)
#                 docs.append(Document(page_content=content, metadata={"source": file.name}))
#             else:
#                 st.warning(f"⚠️ Unsupported file: {file.name}")
#         except Exception as e:
#                 st.error(f"❌ Error loading {file.name}: {e}")
#     return docs

# # --- VECTOR STORE ---
# def build_vector_store(docs):
#     """Documents se ya existing se ek FAISS vector store banata ya update karta hai."""
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     split_docs = splitter.split_documents(docs)
#     try:
#         if os.path.exists(INDEX_PATH):
#             store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
#             store.add_documents(split_docs)
#         else:
#             store = FAISS.from_documents(split_docs, embedding_model)
#         store.save_local(INDEX_PATH)
#         return store
#     except Exception as e:
#         st.warning(f"⚠️ Indexing issue: {e}")
#         return None

# # --- LLM SELECTION ---
# def get_llm():
#     """Initializes and returns the appropriate LLM instance."""
    
#     # Check if a valid OpenAI key has been submitted
#     if st.session_state.get("api_key_submitted") and st.session_state.get("openai_api_key"):
#         openai_model = st.session_state.get("openai_model", "gpt-3.5-turbo")
#         st.sidebar.write(f"✅ Using OpenAI: {openai_model}")
#         return ChatOpenAI(
#             temperature=0.3,
#             openai_api_key=st.session_state['openai_api_key'],
#             model=openai_model
#         )
    
#     else:
#         # Check for Hugging Face token from environment variables
#         try:
#             hf_token = st.secrets["HF_TOKEN"]
#         except KeyError:
#             st.sidebar.error("❌ Error: HF_TOKEN is not configured in `.streamlit/secrets.toml`.")
#             return None

#         # Use the custom LLM class with Hugging Face InferenceClient
#         st.sidebar.write("✅ Using DeepSeek-V3 (HuggingFace API)")
#         client = InferenceClient(token=hf_token)
        
#         # Instantiate the custom LLM
#         return CustomHuggingFaceLLM(
#             client=client,
#             model="deepseek-ai/DeepSeek-V3-0324",
#             max_tokens=512,
#             temperature=0.3
#         )

# # --- CHATBOT INITIALIZATION ---
# def initialize_chatbot():
#     """Chatbot chain ko shuru karta hai agar FAISS index maujood hai."""
#     if not os.path.exists(INDEX_PATH):
#         return None
#     try:
#         store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
#         # LLM instance ko get_llm function se prapt karen
#         llm = get_llm()
#         if llm:
#             # --- Added: Guiding Prompt to focus the model's response ---
#             qa_template = """
#                 You are an AI assistant for a document retrieval system.
#                 Your sole purpose is to answer questions based on the content of the provided documents.
#                 You are not to use any external knowledge. If a question cannot be answered
#                 from the information in the documents, state that you cannot find the answer.
#                 Please provide direct and concise answers.

#                 Chat History:
#                 {chat_history}
#                 Question: {question}
#                 Answer:
#             """
#             qa_prompt = PromptTemplate(input_variables=["question", "chat_history"], template=qa_template)
            
#             return ConversationalRetrievalChain.from_llm(
#                 llm=llm,
#                 retriever=store.as_retriever(),
#                 memory=memory,
#                 chain_type="stuff", # Use "stuff" chain to include the prompt
#                 combine_docs_chain_kwargs={"prompt": qa_prompt} # Pass the custom prompt
#             )
#         else:
#             return None
#     except Exception as e:
#         st.error(f"❌ Chatbot init failed: {e}")
#         return None

# # --- MAIN APP LOGIC ---
# # Sidebar widgets ko load karen aur session state ko shuru karen
# model_choice, uploaded_files, process_button, api_key_valid = sidebar_widgets()
# st.session_state.setdefault("chat_history", [])
# st.session_state.setdefault("documents", [])
# st.session_state.setdefault("model_choice", "Local Model") # Add to session state
# st.session_state.setdefault("qa_chain", None)
# st.session_state.setdefault("openai_model", None)

# # Agar model choice badal gai hai, to session state update karen
# if "last_model_choice" not in st.session_state or st.session_state["last_model_choice"] != model_choice:
#     st.session_state["model_choice"] = model_choice
#     st.session_state["last_model_choice"] = model_choice
#     st.session_state["qa_chain"] = initialize_chatbot() # Re-initialize the chain
#     st.rerun()

# # Document processing
# if uploaded_files and process_button:
#     with st.spinner("🔍 Indexing documents..."):
#         docs = load_documents(uploaded_files)
#         if docs:
#             st.session_state["documents"].extend(docs)
#             build_vector_store(st.session_state["documents"])
#             shutil.rmtree(TEMP_DIR)
#             st.success("✅ Upload Success! Documents indexed and ready for chat.")
#             st.session_state.qa_chain = initialize_chatbot()
#         else:
#             st.warning("⚠️ No valid documents found to process.")

# # Download & clear memory
# if st.session_state.chat_history:
#     try:
#         history_json = json.dumps(clean_chat_history(st.session_state.chat_history), indent=2, ensure_ascii=False)
#         chat_bytes = io.BytesIO(history_json.encode("utf-8"))
#         st.sidebar.download_button("📅 Download Chat History", chat_bytes, "chat_history.json", "application/json")
#     except Exception as e:
#         st.sidebar.error(f"❌ Download failed: {e}")

#     if st.sidebar.button("😹 Clear Memory"):
#         if os.path.exists(INDEX_PATH):
#             shutil.rmtree(INDEX_PATH)
#         st.session_state["documents"] = []
#         st.session_state["chat_history"] = []
#         st.session_state["qa_chain"] = None
#         st.success("✅ Memory and chat history cleared!")

# # --- CUSTOM CSS ---
# st.markdown("""
#     <style>
#     /* Spinner customization */
#     .stSpinner > div > div {
#         border-top-color: #FF69B4; /* Pink */
#         border-right-color: #1E90FF; /* Blue */
#         border-bottom-color: #32CD32; /* Green */
#         border-left-color: #FFD700; /* Gold */
#     }

#     /* Chat bubbles */
#     .user-bubble {
#         background-color: #DCF8C6;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         max-width: 80%;
#         align-self: flex-end;
#         text-align: left;
#     }
#     .bot-bubble {
#         background-color: #E6E6FA;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         max-width: 80%;
#         align-self: flex-start;
#         text-align: left;
#     }

#     /* Colorful words */
#     .thinking-word {
#         color: #FF69B4;
#         font-weight: bold;
#         font-size: 20px;
#     }
#     .query-word {
#         color: #1E90FF;
#         font-weight: bold;
#         font-size: 20px;
#     }
#     .blinking-cursor {
#         animation: blink 1s infinite;
#     }
#     @keyframes blink {
#         0% { opacity: 1; }
#         50% { opacity: 0; }
#         100% { opacity: 1; }
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- CHAT DISPLAY & INPUT ---
# if st.session_state.qa_chain:
#     chat_placeholder = st.container()

#     with chat_placeholder:
#         for msg in st.session_state.chat_history:
#             if msg["role"] == "user":
#                 st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

#     prompt = st.chat_input("Ask your questions here...")

#     if prompt:
#         # User ka message chat history mein jodein
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         st.markdown(
#             f"<span class='thinking-word'>You asked:</span> "
#             f"<span class='query-word'>{prompt}</span>",
#             unsafe_allow_html=True
#         )

#         with st.spinner('Generating...'):
#             try:
#                 # QA chain chalaayein
#                 response = st.session_state.qa_chain.invoke({"question": prompt})
#                 bot_reply = response["answer"] if "answer" in response else "No relevant information found."

#                 with st.chat_message(""):
#                     word_container = st.empty()
#                     full_response = ""
#                     for word in bot_reply.split():
#                         full_response += word + " "
#                         word_container.markdown(f'<div class="bot-bubble">{full_response}<span class="blinking-cursor">|</span></div>', unsafe_allow_html=True)
#                         sleep(0.05)
#                     st.session_state.chat_history.append({"role": "bot", "content": full_response})

#             except Exception as e:
#                 error_message = f"❌ Error: {e}"
#                 st.session_state.chat_history.append({"role": "bot", "content": error_message})

#         st.rerun()

# else:
#     st.info("📄 Please upload and process documents to begin chatting.")








import streamlit as st
import login  # Import login module first

# --- CHECK LOGIN ---
if not st.session_state.get("is_authenticated", False):
    login.login_page()  # Show login page
    st.stop()  # Stop main app until login is done

# --- USER IS LOGGED IN ---
st.title(f"Welcome {st.session_state['user_info'].get('name', '')} to AI Chatbot 🤖")

# --- Your existing main.py code from here ---
# Paste everything from your previous main.py below this login check
# (e.g., sidebar_widgets(), load_documents(), build_vector_store(), get_llm(), chat input/output)
import login
import os
import shutil
import json
import io
from datetime import datetime
import torch
import pandas as pd
import streamlit as st
from time import sleep

# LangChain Imports
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
)
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIApiClient

# Hugging Face Imports for local model
from huggingface_hub import InferenceClient
from typing import Any, List, Optional, Dict
from pydantic import Extra

# --- CONFIGURATION ---
INDEX_PATH = "faiss_index"
TEMP_DIR = "temp_documents"
st.set_page_config(page_title="AI Chatbot for Your Documents", layout="wide")
st.title("AI Chatbot for Your Documents 🤖")

# --- NEW CUSTOM LLM CLASS ---
# This class acts as a bridge between the HuggingFace InferenceClient and LangChain's LLM interface.
# It allows us to use the modern InferenceClient within the LangChain framework.
class CustomHuggingFaceLLM(LLM):
    """A custom LangChain LLM wrapper for the HuggingFace InferenceClient."""
    
    client: InferenceClient
    model: str = "deepseek-ai/DeepSeek-V3-0324"
    max_tokens: int = 512
    temperature: float = 0.3
    
    class Config:
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_client"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """The main method to call the LLM."""
        
        # Use the chat completions endpoint for better performance
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Extract the text from the response
        text = response.choices[0].message.content

        # Handle stop tokens, if any
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        
        return text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

# --- SIDEBAR WIDGETS ---
def sidebar_widgets():
    """Sidebar mein UI elements banata hai jaise ki model selection,
    API key input, aur file uploader.
    """
    st.sidebar.title("⚙️ Settings")
    enable_smart_mode = st.sidebar.checkbox("Enable Smart Thinking Mode", value=False, key="smart_mode_checkbox")

    model_choice = "Open Sorce Model"

    if enable_smart_mode:
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose LLM Model",
            ["Open Sorce Model", "OpenAI better response (paid)"],
            key="llm_model_type_selectbox"
        )

    if model_choice == "OpenAI better response (paid)":
        st.sidebar.subheader("OpenAI better response (paid)")
        openai_model_selected = st.sidebar.selectbox("Choose OpenAI Model", openai_models, key="openai_model_selectbox")
        openai_key = st.sidebar.text_input("Enter OpenAI API Key 🔑", type="password", key="openai_api_key_input")
        submit_key_button = st.sidebar.button("Submit API Key", key="submit_api_key_button")

        if submit_key_button and openai_key:
            st.session_state["openai_api_key"] = openai_key
            try:
                # Key ko validate karne ke liye ek chota sa test call karen
                client = OpenAIApiClient(api_key=openai_key)
                client.models.list()
                st.sidebar.success("✅ API Key is valid!")
                api_key_valid = True
                st.session_state["api_key_submitted"] = True
                st.session_state["openai_model"] = openai_model_selected
            except Exception as e:
                st.sidebar.error(f"❌ Invalid API Key: {e}")
                st.session_state["api_key_submitted"] = False
                api_key_valid = False
                st.session_state["openai_model"] = None
        else:
            st.session_state["openai_model"] = openai_model_selected
    else:
        st.session_state["openai_api_key"] = ""
        st.session_state["api_key_submitted"] = False
        st.session_state["openai_model"] = None
        api_key_valid = True

    st.sidebar.title("📁 Document Input")
    files = st.sidebar.file_uploader("📄 Upload Documents", type=["pdf", "txt", "md", "docx", "csv"], accept_multiple_files=True, key="file_uploader")
    process = st.sidebar.button("🔄 Process & Index Files", key="process_button")
    return model_choice, files, process, api_key_valid

# --- EMBEDDING MODEL ---
@st.cache_resource
def load_embedding_model():
    """HuggingFace embeddings model ko load karta hai aur cache karta hai."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

embedding_model = load_embedding_model()

# --- SANITIZATION & UTILITIES ---
def remove_surrogates(text):
    """Text se surrogate characters hataata hai."""
    try:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore') if isinstance(text, str) else text
    except:
        return ""

def clean_chat_history(chat_history):
    """Chat history se surrogate characters ko saaf karta hai."""
    return [
        {
            "role": remove_surrogates(msg["role"]),
            "content": remove_surrogates(msg["content"]),
            "timestamp": remove_surrogates(msg.get("timestamp", str(datetime.now())))
        }
        for msg in chat_history
    ]

# --- DOCUMENT LOADING ---
def load_documents(files):
    """Uploaded files ko process karta hai aur LangChain Document objects mein load karta hai."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    docs = []
    for file in files:
        path = os.path.join(TEMP_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        ext = os.path.splitext(file.name)[-1].lower()
        try:
            if ext == ".pdf":
                docs.extend(PyPDFLoader(path).load())
            elif ext == ".txt":
                docs.extend(TextLoader(path).load())
            elif ext == ".md":
                docs.extend(UnstructuredMarkdownLoader(path).load())
            elif ext == ".docx":
                docs.extend(UnstructuredWordDocumentLoader(path).load())
            elif ext == ".csv":
                df = pd.read_csv(path, sep=None, engine="python")
                content = df.to_string(index=False)
                docs.append(Document(page_content=content, metadata={"source": file.name}))
            else:
                st.warning(f"⚠️ Unsupported file: {file.name}")
        except Exception as e:
                st.error(f"❌ Error loading {file.name}: {e}")
    return docs

# --- VECTOR STORE ---
def build_vector_store(docs):
    """Documents se ya existing se ek FAISS vector store banata ya update karta hai."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    try:
        if os.path.exists(INDEX_PATH):
            store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
            store.add_documents(split_docs)
        else:
            store = FAISS.from_documents(split_docs, embedding_model)
        store.save_local(INDEX_PATH)
        return store
    except Exception as e:
        st.warning(f"⚠️ Indexing issue: {e}")
        return None

# --- LLM SELECTION ---
def get_llm():
    """Initializes and returns the appropriate LLM instance based on session state."""
    
    if st.session_state.get("model_choice") == "OpenAI better response (paid)":
        if st.session_state.get("api_key_submitted") and st.session_state.get("openai_api_key"):
            openai_model = st.session_state.get("openai_model", "gpt-3.5-turbo")
            st.sidebar.write(f"✅ Using OpenAI: {openai_model}")
            return ChatOpenAI(
                temperature=0.3,
                openai_api_key=st.session_state['openai_api_key'],
                model=openai_model
            )
        else:
            st.sidebar.warning("⚠️ Please submit a valid OpenAI API key.")
            return None
    
    else: # Default to Open Source Model
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except KeyError:
            st.sidebar.error("❌ Error: HF_TOKEN is not configured in `.streamlit/secrets.toml`.")
            return None
        st.sidebar.write("✅ Using DeepSeek-V3 (HuggingFace API)")
        client = InferenceClient(token=hf_token)
        
        return CustomHuggingFaceLLM(
            client=client,
            model="deepseek-ai/DeepSeek-V3-0324",
            max_tokens=512,
            temperature=0.3
        )

# --- CHATBOT INITIALIZATION (Document Q&A specific) ---
def initialize_doc_qa_chain():
    """Chatbot chain ko shuru karta hai agar FAISS index maujood hai."""
    if not os.path.exists(INDEX_PATH):
        return None
    try:
        store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        llm = get_llm()
        if llm:
            qa_template = """
                You are an AI assistant for a document retrieval system.
                Your sole purpose is to answer questions based on the content of the provided documents.
                You are not to use any external knowledge. If a question cannot be answered
                from the information in the documents, state that you cannot find the answer.
                Please provide direct and concise answers.

                Chat History:
                {chat_history}
                Question: {question}
                Answer:
            """
            qa_prompt = PromptTemplate(input_variables=["question", "chat_history"], template=qa_template)
            
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=store.as_retriever(),
                memory=memory,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
        else:
            return None
    except Exception as e:
        st.error(f"❌ Chatbot init failed: {e}")
        return None

# --- MAIN APP LOGIC ---
model_choice, uploaded_files, process_button, api_key_valid = sidebar_widgets()
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("documents", [])
st.session_state.setdefault("model_choice", "Open Sorce Model")
st.session_state.setdefault("doc_qa_chain", None)
st.session_state.setdefault("openai_model", None)

# Update model choice in session state
if "last_model_choice" not in st.session_state or st.session_state["last_model_choice"] != model_choice:
    st.session_state["model_choice"] = model_choice
    st.session_state["last_model_choice"] = model_choice
    st.session_state["doc_qa_chain"] = initialize_doc_qa_chain() if model_choice == "Open Sorce Model" else None
    st.rerun()

# Document processing
if uploaded_files and process_button:
    with st.spinner("🔍 Indexing documents..."):
        docs = load_documents(uploaded_files)
        if docs:
            st.session_state["documents"].extend(docs)
            build_vector_store(st.session_state["documents"])
            shutil.rmtree(TEMP_DIR)
            st.success("✅ Upload Success! Documents indexed and ready for chat.")
            st.session_state.doc_qa_chain = initialize_doc_qa_chain()
        else:
            st.warning("⚠️ No valid documents found to process.")

# Download & clear memory
if st.session_state.chat_history:
    try:
        history_json = json.dumps(clean_chat_history(st.session_state.chat_history), indent=2, ensure_ascii=False)
        chat_bytes = io.BytesIO(history_json.encode("utf-8"))
        st.sidebar.download_button("📅 Download Chat History", chat_bytes, "chat_history.json", "application/json")
    except Exception as e:
        st.sidebar.error(f"❌ Download failed: {e}")

    if st.sidebar.button("😹 Clear Memory"):
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        st.session_state["documents"] = []
        st.session_state["chat_history"] = []
        st.session_state["doc_qa_chain"] = None
        st.success("✅ Memory and chat history cleared!")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Spinner customization */
    .stSpinner > div > div {
        border-top-color: #FF69B4; /* Pink */
        border-right-color: #1E90FF; /* Blue */
        border-bottom-color: #32CD32; /* Green */
        border-left-color: #FFD700; /* Gold */
    }

    /* Chat bubbles */
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-end;
        text-align: left;
    }
    .bot-bubble {
        background-color: #E6E6FA;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-start;
        text-align: left;
    }

    /* Colorful words */
    .thinking-word {
        color: #FF69B4;
        font-weight: bold;
        font-size: 20px;
    }
    .query-word {
        color: #1E90FF;
        font-weight: bold;
        font-size: 20px;
    }
    .blinking-cursor {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- CHAT DISPLAY & INPUT ---
chat_placeholder = st.container()

with chat_placeholder:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

prompt = st.chat_input("Ask your questions here...")

if prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    st.markdown(
        f"<span class='thinking-word'>You asked:</span> "
        f"<span class='query-word'>{prompt}</span>",
        unsafe_allow_html=True
    )

    with st.spinner('Generating...'):
        try:
            bot_reply = "Please upload and process documents to use the chat."
            
            # Use appropriate LLM based on user selection
            if st.session_state["model_choice"] == "OpenAI better response (paid)":
                llm = get_llm()
                if llm:
                    # Direct query to the general LLM
                    response = llm.invoke(prompt)
                    bot_reply = response.content if hasattr(response, 'content') else str(response)
                else:
                    bot_reply = "Please enter a valid OpenAI API key in the sidebar to use this feature."
            
            elif st.session_state["model_choice"] == "Open Sorce Model":
                if st.session_state.doc_qa_chain:
                    # Use the document retrieval chain
                    response = st.session_state.doc_qa_chain.invoke({"question": prompt})
                    bot_reply = response["answer"] if "answer" in response else "I'm sorry, I couldn't find the answer in the provided documents."
                else:
                    bot_reply = "Please upload and process documents to begin chatting in Open Sorce Mode."

            with st.chat_message(""):
                word_container = st.empty()
                full_response = ""
                for word in bot_reply.split():
                    full_response += word + " "
                    word_container.markdown(f'<div class="bot-bubble">{full_response}<span class="blinking-cursor">|</span></div>', unsafe_allow_html=True)
                    sleep(0.05)
                st.session_state.chat_history.append({"role": "bot", "content": full_response})

        except Exception as e:
            error_message = f"❌ Error: {e}"
            st.session_state.chat_history.append({"role": "bot", "content": error_message})

    st.rerun()
else:
    st.info("📄 Please upload and process documents to retrieve the specific knowledge you need.")





