import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Load the pre-trained model from Hugging Face
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("Text to Image Generator")
st.write("This application generates images from text using the Stable Diffusion XL model.")

# Input text from user
prompt = st.text_input("Enter a text prompt to generate an image:")

# Button to generate image
if st.button("Generate Image"):
    if prompt:
        st.write("Generating image...")
        model = load_model()
        with torch.no_grad():
            image = model(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Please enter a text prompt.")
