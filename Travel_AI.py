import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image
import os

# Google API Key from environment variable
# Place your API key here:
os.environ["GOOGLE_API_KEY"] = "AIzaSyB2lBVqSS0t3YzVUghNjhYJZkijvhJaQW8"
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Load the image
try:
    image = Image.open("travel ai.png")  # Replace with your image path
except FileNotFoundError:
    image = None
    st.warning("Please place travel_planner_image.png in the same directory.")

# Title and image
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown(
        '<h1 style="color: white;"><span style="color: cyan;">AI</span>-Powered Travel Planner</h1>',
        unsafe_allow_html=True,
    )
with col2:
    if image:
        st.image(image, width=550)

# Input fields
source = st.text_input("Source", value="Hyderabad")
destination = st.text_input("Destination", value="Pune")

if st.button("Submit"):
    if not google_api_key:
        st.error("Please set the GOOGLE_API_KEY environment variable.")
    else:
        try:
            # Correct model name and potential API version adjustment
            chat_model = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                model="gemini-1.5-pro",  # Or try "gemini-1.5-pro", or "gemini-flash"
                #temperature=0.7,
                # api_version="v1" # Try setting api version if needed, remove if not.
            )
            chat_template = ChatPromptTemplate(
                messages=[
                    ("system", """You are a helpful AI assistant who provides detailed travel 
                    information including modes of transport, estimated costs, and relevant company name . 
                    Giving the  result in the table format, by mode of airways, railway, etc.."""),
                    
                    ("human", "Provide travel information from {source} to {destination}, including available modes of transport, estimated costs, and logos for each mode.")
                ],
                partial_variables={}
            )
            parser = StrOutputParser()
            chain = chat_template | chat_model | parser
            raw_input = {"source": source, "destination": destination}
            result = chain.invoke(raw_input)
            st.write(result)

            # Example logo handling (adjust based on Gemini output)
            if "flight" in result.lower():
                try:
                    flight_logo = Image.open("flight_logo.png")
                    st.image(flight_logo, width=50)
                except FileNotFoundError:
                    st.write("Flight logo not found")
            if "train" in result.lower():
                try:
                    train_logo = Image.open("train_logo.png")
                    st.image(train_logo, width=50)
                except FileNotFoundError:
                    st.write("Train logo not found")
            # Add similar logic for other modes of transport

        except Exception as e:
            st.error(f"An error occurred: {e}")