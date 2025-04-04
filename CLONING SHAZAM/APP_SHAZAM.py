import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import re
import os
import whisper
import tempfile
import moviepy.editor as mp

# Streamlit app title and description
st.title("🎵 Shazam Clone: Audio/Video Subtitle Search")
st.markdown("""
    Upload an audio or video file, and this app will identify spoken content and find matching subtitle chunks from our database.  
    This app mimics Shazam by identifying audio content and linking it to relevant subtitles.
""")

# Step 1: Initialize Model and ChromaDB Collection (Cached for Performance)
@st.cache_resource
def load_model_and_collection():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        # In Hugging Face Spaces, chroma.sqlite3 is in the same directory as this script
        persist_path = os.path.dirname(os.path.abspath(__file__))
        
        # Check if chroma.sqlite3 exists
        db_path = os.path.join(persist_path, "chroma.sqlite3")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB database file not found at: {db_path}")
        
        client = chromadb.PersistentClient(path=persist_path)
        # Verify the collection exists
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        if "subtitle_embeddings" not in collection_names:
            raise ValueError(f"Collection 'subtitle_embeddings' not found in ChromaDB. Available collections: {collection_names}")
        
        collection = client.get_collection("subtitle_embeddings")
        return model, collection, persist_path
    except Exception as e:
        raise Exception(f"Error loading model or collection: {str(e)}")

try:
    model, collection, persist_path = load_model_and_collection()
    st.success(f"Loaded collection 'subtitle_embeddings' from: {persist_path}")
    st.write(f"Total chunks in collection: {collection.count()}")
except Exception as e:
    st.error(f"Failed to load collection: {e}")
    st.stop()

# Step 2: Query Preprocessing Function
def preprocess_query(query):
    """Clean and preprocess the query to match subtitle cleaning."""
    query = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', '', query)
    query = re.sub(r'^\d+\r?\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'\s+', ' ', query.strip())
    query = re.sub(r'\[.*?\]|\(.*?\)', '', query)
    query = re.sub(r'<.*?>', '', query)
    return query.lower()

# Step 3: Speech-to-Text Function using Whisper
@st.cache_resource
def load_whisper_model():
    try:
        if not hasattr(whisper, 'load_model'):
            raise AttributeError("The 'whisper' module does not have a 'load_model' function. Ensure you have installed 'openai-whisper', not the 'whisper' package.")
        return whisper.load_model("base")  # Use the 'base' model for better performance on CPU
    except Exception as e:
        raise Exception(f"Failed to load Whisper model: {str(e)}")

def extract_audio_from_video(video_file):
    """Extract audio from a video file and save as a temporary WAV file."""
    try:
        # Save the video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_path = temp_video_file.name

        # Extract audio using moviepy
        video = mp.VideoFileClip(temp_video_path)
        audio = video.audio
        temp_audio_path = temp_video_path.replace(".mp4", ".wav")
        audio.write_audiofile(temp_audio_path)
        audio.close()
        video.close()
        
        # Clean up the temporary video file
        os.unlink(temp_video_path)
        return temp_audio_path
    except Exception as e:
        raise Exception(f"Error extracting audio from video: {e}")

def audio_to_text(audio_file, file_type):
    """Convert audio file (or audio extracted from video) to text using Whisper."""
    try:
        # Load Whisper model
        whisper_model = load_whisper_model()
        
        # If the file is a video, extract audio first
        if file_type in ["video/mp4"]:
            temp_audio_path = extract_audio_from_video(audio_file)
        else:
            # Save the audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_file.read())
                temp_audio_path = temp_file.name

        # Transcribe the audio
        result = whisper_model.transcribe(temp_audio_path, fp16=False)  # Disable FP16 for CPU compatibility
        text = result["text"].strip()
        
        # Clean up the temporary audio file
        os.unlink(temp_audio_path)
        
        # Check if the transcription is empty or meaningless
        if not text or text.isspace():
            return "Transcription resulted in empty text. The audio may not contain recognizable speech or lyrics."
        return text
    except Exception as e:
        return f"Error processing audio with Whisper: {str(e)}"
    finally:
        # Ensure the temporary file is deleted even if an error occurs
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

# Step 4: Search Function
def search_subtitles(query, top_k=5):
    """Retrieve top_k subtitle chunks matching the query using semantic search."""
    try:
        cleaned_query = preprocess_query(query)
        query_embedding = model.encode([cleaned_query], show_progress_bar=False)[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        retrieved_results = []
        for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            retrieved_results.append({
                'Num': metadata['num'],
                'Name': metadata['name'],
                'Subtitle Chunk': doc,
                'Similarity Score': round(1 - distance, 3)
            })
        return pd.DataFrame(retrieved_results)
    except Exception as e:
        st.error(f"Error during search: {e}")
        return pd.DataFrame()

# Step 5: File Upload and Processing
st.header("📤 Upload Audio or Video File")
with st.container():
    uploaded_file = st.file_uploader(
        "Choose an audio (.mp3, .wav) or video (.mp4) file",
        type=['mp3', 'wav', 'mp4'],
        help="Upload an audio or video file to identify spoken content and find matching subtitles."
    )

if uploaded_file is not None:
    # Display the uploaded file
    st.header("📄 Uploaded File")
    with st.container():
        st.write(f"**File Name**: {uploaded_file.name}")
        st.write(f"**File Type**: {uploaded_file.type}")
        
        # Save file temporarily for processing
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved temporarily at: {file_path}")
            
            # Display the file for playback
            if uploaded_file.type in ["audio/mpeg", "audio/wav"]:
                st.audio(uploaded_file)
            elif uploaded_file.type == "video/mp4":
                st.video(uploaded_file)
        except Exception as e:
            st.error(f"Failed to save or display file: {e}")
            st.stop()

    # Extract text from audio
    st.header("🗣️ Extracted Text from Audio")
    with st.container():
        with st.spinner("Extracting spoken content from audio..."):
            extracted_text = audio_to_text(uploaded_file, uploaded_file.type)
        
        # Display extracted text in a styled box
        if extracted_text and not extracted_text.startswith("Error processing audio") and not extracted_text.startswith("Transcription resulted in empty text"):
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db;">
                    <p style="font-size: 16px; color: #333;">{extracted_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error(f"Failed to extract meaningful text from the audio: {extracted_text}")
            st.info("This might happen if the audio is a song with heavy instrumentals, background noise, or unclear vocals. Try uploading an audio or video file with clear speech (e.g., a podcast, narration, or spoken dialogue).")
            st.stop()

    # Search for matching subtitles
    st.header("🔍 Search Results")
    with st.container():
        with st.spinner("Searching for matching subtitles..."):
            results = search_subtitles(extracted_text, top_k=5)
        
        if not results.empty:
            st.write("Here are the top matching subtitle chunks from the database:")
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No matching subtitle chunks found for the extracted text.")

# Step 6: Additional Features
st.sidebar.header("⚙️ Options")
top_k = st.sidebar.slider("Number of results to display", min_value=1, max_value=10, value=5)
st.sidebar.markdown("Adjust the number of subtitle chunks returned by the search.")

# Footer
st.markdown("---")
st.write("Built with Streamlit | Project: Shazam Clone for Subtitle Search")