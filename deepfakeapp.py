import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import os

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load('deepfake_resnet50.pth', map_location='cpu'))
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Extract frames
def extract_frames(video_path, frame_rate=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image)
        frame_count += 1
    cap.release()
    return frames

# Classify video and collect fake frames
def test_video(video_path, model):
    frames = extract_frames(video_path, frame_rate=10)
    if not frames:
        return "No frames found.", 0, 0, 0, []

    fake_frames = []
    fake_count = 0

    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 1:
                fake_count += 1
                fake_frames.append(frame)

    total_frames = len(frames)
    fake_percent = (fake_count / total_frames) * 100
    result = "Fake" if fake_count > total_frames / 2 else "Real"
    return result, total_frames, fake_count, fake_percent, fake_frames

# --- Streamlit UI ---

st.markdown("""
<style>
    @keyframes shine {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    .attractive-title {
        text-align: center;
        font-size: 3.5em;
        font-weight: 800;
        background: linear-gradient(45deg, #8A2BE2, #FF10F0, #8A2BE2);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
        margin-bottom: 0;  /* Remove default margin */
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #8A2BE2;
        opacity: 0.8;
        margin-top: -10px;  /* Pull closer to title */
        font-style: italic;
    }
</style>
<h1 class="attractive-title">✨ DeepFake Detector ✨</h1>
<p class="subtitle">by Shruti Hulke</p>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    video_path = os.path.join("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Classifying...")
    model = load_model()
    result, total_frames, fake_count, fake_percent, fake_frames = test_video(video_path, model)

    st.success(f"Video classified as: **{result}**")
    st.info(f"Total extracted frames: {total_frames}")
    if result == "Fake":
        st.info(f"Fake frames detected: {fake_count}")
        st.info(f"Percentage of fake frames: {fake_percent:.2f}%")

    else:
        st.warning("No fake frames detected.")
