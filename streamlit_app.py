import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask_stream"

st.set_page_config(
    page_title="TriMind",
    layout="wide"
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("🧠 TriMind")

    st.session_state.dark_mode = st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode
    )

    if st.button("➕ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat = chat_id
        st.session_state.chats[chat_id] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Chats")

    for chat_id in list(st.session_state.chats.keys()):

        col1, col2 = st.columns([4,1])

        with col1:
            if st.button(
                f"Chat {chat_id[:4]}",
                key=f"select_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat = chat_id
                st.rerun()

        with col2:
            if st.button(
                "🗑",
                key=f"delete_{chat_id}",
                use_container_width=True
            ):
                del st.session_state.chats[chat_id]

                if not st.session_state.chats:
                    new_id = str(uuid.uuid4())
                    st.session_state.chats[new_id] = []
                    st.session_state.current_chat = new_id
                else:
                    st.session_state.current_chat = list(
                        st.session_state.chats.keys()
                    )[0]

                st.rerun()

# -----------------------------
# THEME CSS
# -----------------------------
if st.session_state.dark_mode:

    st.markdown("""
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #0E1117;
    }

    [data-testid="stChatMessage"] {
        background-color: #1E222A;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }

    textarea {
        background-color: #1E222A !important;
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    .stApp {
        background-color: #FFFFFF;
        color: #111111;
    }

    section[data-testid="stSidebar"] {
        background-color: #F5F7FA;
    }

    [data-testid="stChatMessage"] {
        background-color: #F1F3F6;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        color: black;
    }

    textarea {
        background-color: white !important;
        color: black !important;
        border: 1px solid #d1d5db;
    }

    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# CENTERED LAYOUT
# -----------------------------
st.markdown(
"""
<style>
.block-container {
    max-width: 850px;
    margin: auto;
}
</style>
""",
unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
"""
<h1 style='text-align:center;'>TriMind</h1>
<h4 style='text-align:center; color:gray; margin-top:-10px'>
RAG Powered Knowledge Assistant
</h4>
""",
unsafe_allow_html=True
)

# -----------------------------
# CHAT HISTORY
# -----------------------------
chat_id = st.session_state.current_chat
messages = st.session_state.chats[chat_id]

if len(messages) == 0:
    st.info("👋 Hello! Ask anything about your knowledge base.")

for role, content in messages:

    with st.chat_message(role):
        st.markdown(content)

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:

    messages.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        try:

            response = requests.post(
                API_URL,
                json={"question": user_input},
                stream=True,
                timeout=60
            )

            for chunk in response.iter_content(chunk_size=10):

                if chunk:
                    text = chunk.decode("utf-8")
                    full_response += text
                    placeholder.markdown(full_response)

        except Exception:

            full_response = (
                "⚠️ Currently we are unable to connect to the AI service.\n\n"
                "Please try again in a moment."
            )

            placeholder.markdown(full_response)

    messages.append(("assistant", full_response))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("TriMind • Retrieval Augmented Generation AI Assistant")