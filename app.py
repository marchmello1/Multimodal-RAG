import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from gtts import gTTS
import torch
import torchaudio
import soundfile as sf
from io import BytesIO
from together import Together
import requests
import base64

# Set the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["TOGETHER_API_KEY"]
client = Together(api_key=api_key)

# Set the Hugging Face API URL and headers
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['api_keys']['HUGGINGFACE_API_KEY']}"}

# Load smaller text model
@st.cache_resource
def load_text_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load audio models on demand
def load_audio_models():
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return audio_processor, audio_model

# Extract text and images from PDF
def extract_text_and_images(pdf_file):
    pdf_data = pdf_file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    texts = []
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texts.append(page.get_text("text"))

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(BytesIO(image_bytes))

    return texts, images

# Generate summaries using Mistral model
def generate_mistral_summary(text, max_length=150):
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[{"role": "user", "content": f"Please summarize the following text: {text}"}],
            max_tokens=max_length
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Summary generation failed."

def query_hf_image_model(image):
    img_bytes = image.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "inputs": img_base64
    }
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    return response.json()

def transcribe_audio(audio_file):
    audio_processor, audio_model = load_audio_models()
    audio_data, sample_rate = sf.read(audio_file)
    
    if sample_rate != 16000:
        # Resample the audio to 16 kHz
        audio_data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(torch.tensor(audio_data).float())
        sample_rate = 16000
    
    input_values = audio_processor(audio_data, return_tensors="pt", sampling_rate=sample_rate).input_values
    logits = audio_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = audio_processor.decode(predicted_ids[0])
    return transcription

def text_to_audio(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)
    return output_path

def multimodal_query(query, query_type='text', k=3):
    if query_type == 'text':
        text_model = load_text_model()
        retrieved_texts = retrieve_text(query, k)
        summarized_texts = [generate_mistral_summary(text, max_length=min(150, len(text)//2)) for text in retrieved_texts if text.strip()]
        combined_texts = " ".join(summarized_texts) if summarized_texts else "No valid text summaries available."
        final_response = generate_mistral_summary(combined_texts, max_length=min(300, len(combined_texts)//2)) if summarized_texts else combined_texts
    elif query_type == 'image':
        retrieved_images = retrieve_images(query, k)
        image_captions = [query_hf_image_model(img)['generated_text'] for img in retrieved_images]
        final_response = " ".join(image_captions)
    elif query_type == 'audio':
        transcription = transcribe_audio(query)
        final_response = generate_mistral_summary(transcription, max_length=min(300, len(transcription)//2))
    else:
        final_response = "Unsupported query type."

    audio_output_path = text_to_audio(final_response, "output_audio.mp3")

    return final_response, audio_output_path

# Retrieve text
def retrieve_text(query, k=10):
    text_model = load_text_model()
    query_embedding = text_model.encode([query], convert_to_tensor=False)
    distances, indices = text_index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Retrieve images
def retrieve_images(query, k=10):
    return images[:k]

# Streamlit app
st.title("Multimodal RAG System")

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["pdf", "jpg", "jpeg", "png", "wav", "mp3"], accept_multiple_files=True)

# Containers for text, images, and audio data
texts = []
images = []
audios = []

# Process uploaded files
for uploaded_file in uploaded_files:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'application' and uploaded_file.type.split('/')[1] == 'pdf':
        pdf_texts, pdf_images = extract_text_and_images(uploaded_file)
        texts.extend(pdf_texts)
        images.extend(pdf_images)
    elif file_type == 'image':
        images.append(uploaded_file)
    elif file_type == 'audio':
        audios.append(uploaded_file)

# Compute embeddings if there are any texts
if texts:
    text_model = load_text_model()
    text_embeddings = text_model.encode(texts, convert_to_tensor=False)
    dimension_text = text_embeddings.shape[1]
    text_index = faiss.IndexFlatL2(dimension_text)
    text_index.add(text_embeddings)

# Query input
query = st.text_input("Ask a question about the uploaded files")

if st.button("Submit Query"):
    if query:
        if texts:
            response, audio_output = multimodal_query(query, query_type='text')
        elif images:
            response, audio_output = multimodal_query(query, query_type='image')
        elif audios:
            transcription = transcribe_audio(audios[0])
            response, audio_output = multimodal_query(transcription, query_type='audio')
            st.write("Transcription:", transcription)
        
        st.write("Response:", response)
        st.audio(audio_output)
        if images:
            for img in images:
                img_caption = query_hf_image_model(img)['generated_text']
                st.image(img, caption=img_caption)
