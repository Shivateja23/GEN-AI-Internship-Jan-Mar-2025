import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Streamlit UI
st.title("AI Conversational Data Science Tutor (Gemini 1.5 Pro)")

# Google API Key (using Streamlit secrets)

try:
    GOOGLE_API_KEY = "AIzaSyCn8KT7AuYQpZmQ8VEUA7oeKmqMVQKZqXo"  # Correct key name
except KeyError:
    st.error(
        "Google API key not found in secrets.toml. Please ensure you have created the secrets.toml file and added your API key with the correct name: GOOGLE_API_KEY"
    )
    st.stop()  # stop the app if the key is not found.

# Initialize ChatGoogleGenerativeAI model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Initialize Conversation Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Constructing Prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a friendly and helpful AI Tutor with expertise in Data Science and AI. You will provide clear explanations, code examples, and answer questions to help the user learn. You will maintain the context of the conversation."""
)
human_prompt = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, HumanMessagePromptTemplate.from_template("{history}"), human_prompt]
)

# Chain
chain = (
    {
        "input": RunnablePassthrough(),
        "history": lambda x: st.session_state.memory.load_memory_variables({})["history"],
    }
    | chat_prompt
    | chat_model
    | StrOutputParser()
)

# User input
user_input = st.text_input("Ask your Data Science question:")

if user_input:
    try:
        output = chain.invoke(user_input)
        st.session_state.memory.save_context({"input": user_input}, {"output": output})
        st.write("AI Tutor:")
        st.write(output)

        # Display conversation history
        with st.expander("Conversation History"):
            messages = st.session_state.memory.load_memory_variables({})["history"]
            for message in messages:
                if message.type == "human":
                    st.write(f"User: {message.content}")
                else:
                    st.write(f"AI: {message.content}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

