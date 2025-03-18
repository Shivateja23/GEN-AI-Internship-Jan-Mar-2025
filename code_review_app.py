import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the API key from .python_app_env
load_dotenv(".python_app_env")
api_key = os.getenv("GOOGLE_API_KEY")  # Fetch the API key

# Configure the Gemini API
genai.configure(api_key=api_key)

def review_code(code):
    """Send the user's Python code to Gemini API for review and fixes."""

    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Generate response from Gemini
        response = model.generate_content(f"""
        You are a Python code reviewer. Analyze the given code for errors, bugs, and improvements.
        Provide a list of identified issues, along with explanations and suggestions for fixes.
        Then, provide the corrected version of the code.

        Code:
        ```python
        {code}
        ```

        Output format:
        Issues:
        - Issue 1: Explanation and suggested fix.
        - Issue 2: Explanation and suggested fix.

        Fixed Code:
        ```python
        (Corrected version of the code here)
        ```
        """)

        return response.text  # Extract the response text
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    #st.title("Python Code Reviewer Chatbot")
    st.markdown(
    '<h1 style="text-align: center;">'
    '<span style="color: skyblue;">Python Code</span>  '
    '<span style="color: white;">Reviewer </span>  '
    '<span style="color: yellow;">Chatbot</span>  '
    '</h1>',
    unsafe_allow_html=True
)

    #st.image("python img.png", width=300)
    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns
    with col2:  # Place the image in the middle column
      st.image("python img.png", width=300)


    st.write("Submit your Python code below, and receive a review with bug fixes!")

    user_code = st.text_area("Enter your Python code here:", height=300)

    if st.button("Review Code"):
        if user_code.strip():
            st.write("## Review Results:")
            feedback = review_code(user_code)
            st.markdown(feedback)
        else:
            st.warning("Please enter some Python code before submitting.")

if __name__ == "__main__":
    main()

