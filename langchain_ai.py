import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Streamlit UI
st.title("Data Science Python Tutor (Gemini)")

# Google API Key (using Streamlit secrets)

try:
    GOOGLE_API_KEY = "AIzaSyCn8KT7AuYQpZmQ8VEUA7oeKmqMVQKZqXo"  # Correct key name
except KeyError:
    st.error("Google API key not found in secrets.toml. Please ensure you have created the secrets.toml file and added your API key with the correct name: GOOGLE_API_KEY")
    st.stop() # stop the app if the key is not found.

# Initialize ChatGoogleGenerativeAI model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Constructing Prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a friendly AI Tutor with expertise in Data Science and AI who tells step by step Python Implementation for topics asked by user."""
)
human_prompt = HumanMessagePromptTemplate.from_template(
    "Tell me a python implementation for {topic_name}."
)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Output Parser
output_parser = StrOutputParser()

# Chain
chain = chat_prompt | chat_model | output_parser

# User input
topic_name = st.text_input("Enter the Data Science topic:")

if topic_name:
    raw_input = {"topic_name": topic_name}
    try:
        output = chain.invoke(raw_input)
        st.write("Python Implementation:")
        st.code(output, language="python")  # Display as code
    except Exception as e:
        st.error(f"An error occurred: {e}")