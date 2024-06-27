import streamlit as st
import fitz 
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
from transformers import BlipProcessor, BlipForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
from together import Together
from gtts import gTTS
import torch
import soundfile as sf
from io import BytesIO

# Set the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["TOGETHER_API_KEY"]
client = Together(api_key=api_key)

# Function to load models on demand
@st.cache_resource
def load_text_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_image_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_resource
def load_audio_models():
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return audio_processor, audio_model

@st.cache_resource
def load_blip_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model

# Function to extract text and images from PDF
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

# Function to generate summaries using Together API
def generate_mistral_summary(text, max_length=150):
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[{"role": "user", "content": f"Please summarize the following text: {text}"}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Summary generation failed."

def generate_image_caption(image):
    blip_processor, blip_model = load_blip_models()
    try:
        inputs = blip_processor(image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return "Caption generation failed."

def transcribe_audio(audio_file):
    audio_processor, audio_model = load_audio_models()
    audio_data, sample_rate = sf.read(audio_file)
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
    text_model = load_text_model()
    if query_type == 'text':
        retrieved_texts = retrieve_text(query, k)
        summarized_texts = [generate_mistral_summary(text) for text in retrieved_texts if text.strip()]
        combined_texts = " ".join(summarized_texts) if summarized_texts else "No valid text summaries available."
        final_response = generate_mistral_summary(combined_texts, max_length=300) if summarized_texts else combined_texts

    elif query_type == 'image':
        image_model = load_image_model()
        retrieved_images = retrieve_images(query, k)
        image_captions = [generate_image_caption(img) for img in retrieved_images]
        final_response = " ".join(image_captions)

    elif query_type == 'audio':
        transcription = transcribe_audio(query)
        final_response = generate_mistral_summary(transcription, max_length=300)

    audio_output_path = text_to_audio(final_response, "output_audio.mp3")

    return final_response, audio_output_path

# Function to retrieve text
def retrieve_text(query, k=10):
    query_embedding = text_model.encode([query], convert_to_tensor=False)
    distances, indices = text_index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Function to retrieve images
def retrieve_images(query, k=10):
    query_embedding = image_model.encode([query], convert_to_tensor=False)
    distances, indices = image_index.search(query_embedding, k)
    return [images[i] for i in indices[0]]

# Streamlit app
st.title("Multimodal RAG System")

# Unified input
input_type = st.selectbox("Select input type", ["Text", "Image", "Audio", "PDF"])
input_data = None

if input_type == "Text":
    input_data = st.text_area("Enter text")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        input_data = Image.open(uploaded_image)

elif input_type == "Audio":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_audio:
        input_data = BytesIO(uploaded_audio.read())

elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        texts, images = extract_text_and_images(uploaded_pdf)
        st.success("PDF processed successfully!")
        input_data = {"texts": texts, "images": images}

if st.button("Submit"):
    if input_type == "Text":
        query = input_data
        response, audio_output = multimodal_query(query, query_type='text')
        st.write("Response:", response)
        st.audio(audio_output)

    elif input_type == "Image":
        image = input_data
        caption = generate_image_caption(image)
        st.image(image, caption=caption)

    elif input_type == "Audio":
        transcription = transcribe_audio(input_data)
        response, audio_output = multimodal_query(transcription, query_type='text')
        st.write("Transcription:", transcription)
        st.write("Response:", response)
        st.audio(audio_output)

    elif input_type == "PDF":
        texts = input_data["texts"]
        images = input_data["images"]
        
        text_model = load_text_model()
        text_embeddings = text_model.encode(texts, convert_to_tensor=False)
        dimension_text = text_embeddings.shape[1]
        text_index = faiss.IndexFlatL2(dimension_text)
        text_index.add(text_embeddings)

        image_model = load_image_model()
        image_embeddings = []
        for img in images:
            img = Image.open(img)
            img_emb = image_model.encode(img, convert_to_tensor=False)
            image_embeddings.append(img_emb)
        
        if image_embeddings:
            image_embeddings = np.array(image_embeddings)
            dimension_image = image_embeddings.shape[1]
            image_index = faiss.IndexFlatL2(dimension_image)
            image_index.add(image_embeddings)
        else:
            st.warning("No valid image embeddings were created.")

        query = st.text_input("Enter your query for the PDF content")
        if st.button("Query PDF"):
            response, audio_output = multimodal_query(query, query_type='text')
            st.write("Response:", response)
            st.audio(audio_output)
