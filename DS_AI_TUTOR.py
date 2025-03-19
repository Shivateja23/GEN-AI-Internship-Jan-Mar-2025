import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load API Key from .DS_AI_tutore_env file
env_loaded = load_dotenv(".DS_AI_tutore_env")  # Explicitly load the env file

api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is loaded correctly
if not api_key:
    st.error("⚠️ Google API key is missing! Please check your .DS_AI_tutore_env file.")
    st.stop()

# Set API Key in Environment Variables
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Chat Model with Error Handling
try:
    chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7, google_api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize AI model: {e}")
    st.stop()

# Initialize Memory for Conversation History
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Streamlit UI Configuration
st.set_page_config(page_title='AI Data Science Tutor', page_icon="📊", layout='wide')

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .stApp {
            background: #D3D3D3; /* Gray Ash */
            color: #333;
            font-family: 'Poppins', sans-serif;
        }
        [data-testid="stSidebar"] {
            background: #2C3E50; /* Dark Gray */
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        h1 {
            text-align: center;
            color: #1F2937;
            font-weight: 600;
        }
        .stButton > button {
            background: linear-gradient(to right, #ff758c, #ff7eb3);
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background: linear-gradient(to right, #ff7eb3, #ff758c);
        }
        .chat-container {
            max-height: 550px;
            overflow-y: auto;
            padding: 15px;
            border-radius: 15px;
            background: #f7f7f7;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.15);
        }
        .message-box {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .user-message {
            background-color: #4CAF50;
            color: white;
            align-self: flex-end;
            text-align: right;
        }
        .ai-message {
            background-color: #e3e3e3;
            color: black;
            align-self: flex-start;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - About the App
st.sidebar.title("About this App ❓")
st.sidebar.info("""
🚀 **Learn Data Science with AI**
- Ask any **Data Science** questions
- Get structured explanations **based on your level**
- Interactive, simple, and effective learning

**Enjoy AI-powered tutoring!**
""")

# Sidebar - Learning Level Selection
st.sidebar.title(" Settings")
user_level = st.sidebar.radio("Select your learning level:", ["Beginner", "Intermediate", "Advanced"], horizontal=True)

# System Message for AI
system_message = SystemMessage(
    content=f"You are an AI tutor specialized in Data Science. Provide responses based on the user's level: {user_level}."
)

# Main Title
st.title("🤖 AI-Powered Data Science Tutor")

# Chat Container
st.markdown("### 💬 Conversation")
chat_container = st.container()

# Display Chat History
with chat_container:
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f'<div class="message-box user-message"><b>You:</b> {msg.content}</div>', unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f'<div class="message-box ai-message"><b>AI:</b> {msg.content}</div>', unsafe_allow_html=True)

# User Input
user_query = st.chat_input("Ask me anything about Data Science...")
if user_query:
    conversation_history = [system_message] + st.session_state.memory.chat_memory.messages + [HumanMessage(content=user_query)]
    
    try:
        ai_response = chat_model.invoke(conversation_history)
    except Exception as e:
        st.error(f"❌ AI response error: {e}")
        st.stop()

    # Store messages in memory
    st.session_state.memory.chat_memory.add_user_message(user_query)
    st.session_state.memory.chat_memory.add_ai_message(ai_response.content)

    # Display Messages
    with chat_container:
        st.markdown(f'<div class="message-box user-message"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message-box ai-message"><b>AI:</b> {ai_response.content}</div>', unsafe_allow_html=True)

# Sidebar - Chat History Download
if st.session_state.memory.chat_memory.messages:
    chat_text = "\n".join([f"You: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in st.session_state.memory.chat_memory.messages])
    st.sidebar.download_button("📥 Download Chat History", chat_text, file_name="chat_history.txt")

# Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.memory.clear()
    st.rerun()

