import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".code_reviewer_env")
api_key = os.getenv("GOOGLE_API_KEY")

# App Config
st.set_page_config(page_title="AI Python Code Reviewer", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #00BFFF;'>🤖 AI-Powered Python Code Reviewer</h1>",
    unsafe_allow_html=True
)

st.markdown("##### Paste your Python code or upload a `.py` file for instant review.")

# Input: Text area
user_code = st.text_area("📝 Paste your Python code here:", height=300, placeholder="# Paste or upload code...")

# Input: File uploader
uploaded_file = st.file_uploader("📂 Or upload a Python (.py) file", type=["py"])
if uploaded_file is not None:
    try:
        user_code = uploaded_file.read().decode("utf-8")
        st.success("✅ File uploaded successfully. Ready to review.")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Submit button
if st.button("🚀 Review My Code"):
    if not api_key:
        st.error("🔑 Please set the GOOGLE_API_KEY in your .code_reviewer_env file.")
    elif not user_code.strip():
        st.warning("⚠️ Please provide some code to review.")
    else:
        try:
            # Initialize Gemini Model
            model = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-1.5-pro"
            )

            # Review Prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert Python code reviewer. Carefully review the given Python code.

Provide:
- A list of bugs or potential issues.
- Suggestions for improvements and best practices.
- Performance optimizations.
- Security concerns, if any.
- Constructive, beginner-friendly feedback.

Output the review in markdown bullet points with clear structure. Do not include the original code in your response."""),
                ("human", "Here is the Python code:\n\n{code}")
            ])

            # Output pipeline
            parser = StrOutputParser()
            chain = prompt | model | parser

            # Invoke model
            result = chain.invoke({"code": user_code})

            # Show output
            st.markdown("### 🧠 Code Review Feedback:")
            st.markdown(result)

        except Exception as e:
            st.error(f"❌ An error occurred while reviewing: {e}")
