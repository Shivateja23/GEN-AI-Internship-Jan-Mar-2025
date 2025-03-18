import streamlit as st
import google.generativeai as ai

from PIL import Image
# Set your Gemini API key
ai.configure(api_key="AIzaSyBkODCV5B1C9V8117nG7UnaSbqB4HVjAQ0")  # Replace with your actual key

# Initialize Gemini model for code review
model = ai.GenerativeModel(model_name="models/gemini-pro")  # Or your preferred Gemini model

# Streamlit app
st.title("Python Code Reviewer App")
#st.image("python img.png", width=200, height=150)#

img = Image.open("python img.png")  # Open the image using PIL
new_img = img.resize((200, 150))  # Resize to 200x150 pixels (width, height)
st.image(new_img)  # Display the resized image

# Input method selection
input_method = st.radio("Select Input Method:", ("File Upload", "Paste Code"))

code = ""  # Initialize code variable

if input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload a Python file", type="py")
    if uploaded_file is not None:
        code = uploaded_file.read().decode("utf-8")  # Decode file content
elif input_method == "Paste Code":
    code = st.text_area("Paste your Python code here", height=300)

if code:  # Proceed only if code is provided
    if st.button("Review Code"):
        # Construct the prompt for code review
        prompt = f"""
        Review the following Python code and provide feedback on:
        - Potential errors and bugs
        - Code style and best practices
        - Suggestions for improvements and optimizations
        - Security vulnerabilities (if applicable)
        - give the correcet code

        ```python
        {code}
        ```
        """

        # Generate the review using Gemini
        response = model.generate_content(prompt)
        review = response.text

        # Display the review
        st.subheader("Code Review")
        st.write(review)