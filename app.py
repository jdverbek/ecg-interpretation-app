import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import io

# Load the pre-trained model and processor from Hugging Face
@st.cache_resource  # Cache for performance
def load_model():
    processor = AutoImageProcessor.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")
    model = AutoModelForImageClassification.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")
    return processor, model

processor, model = load_model()

# Define label mapping based on model/dataset (alphabetical order: F, M, N, Q, S, V)
id2label = {
    0: "Fusion of ventricular and normal beat",
    1: "Myocardial Infarction",
    2: "Normal beat",
    3: "Unclassifiable beat",
    4: "Supraventricular premature beat",
    5: "Premature ventricular contraction"
}

# Streamlit app interface
st.title("ECG Interpretation App")
st.write("Upload a photo of an ECG (taken with your phone) for classification. Note: This is for educational purposes only—not a substitute for professional medical advice.")

uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference (on CPU)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = id2label.get(predicted_class_idx, "Unknown")
    
    st.write(f"**Predicted Interpretation:** {predicted_label}")
    st.write("Probabilities (for all classes):")
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    for idx, prob in enumerate(probs):
        st.write(f"{id2label.get(idx, 'Unknown')}: {prob:.2%}")

