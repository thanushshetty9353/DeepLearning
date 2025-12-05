import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

st.set_page_config(page_title="Digit Predictor")

st.title("🖤 AI Blackboard — Draw a Digit")

canvas = st_canvas(
    stroke_width=14,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

predict_btn = st.button("🔮 Predict Digit")

if predict_btn and canvas.image_data is not None:
    img = canvas.image_data.astype("uint8")
    pil = Image.fromarray(img).convert("L")
    pil = pil.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = F.softmax(out, dim=1).cpu().numpy()[0]
        digit = int(np.argmax(prob))
        conf = float(prob[digit]) * 100

    st.success(f"Predicted Digit: **{digit}**")
    st.info(f"Confidence: **{conf:.2f}%** ⭐")
