import streamlit as st
st.set_page_config(page_title="Digit Predictor", layout="wide")

from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go

# ----------------------- MODEL -----------------------
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


# ----------------------- CUSTOM CSS -----------------------
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        color: #ffffff;
        background: linear-gradient(90deg, #7928CA, #FF0080);
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 25px;
    }
    .glass-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 18px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)



# ----------------------- HEADER -----------------------
st.markdown("<div class='title'>🖤 AI Blackboard — Draw a Digit</div>", unsafe_allow_html=True)


col1, col2 = st.columns([1,1])

# ----------------------- CANVAS -----------------------
with col1:
    st.markdown("### ✏️ Draw Here")
    with st.container():
        canvas = st_canvas(
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    predict_btn = st.button(" Predict Digit", use_container_width=True)
    clear_btn = st.button("🧹 Clear Board", use_container_width=True)

    if clear_btn:
        st.experimental_rerun()


# ----------------------- RESULT SECTION -----------------------
with col2:
    st.markdown("### 📊 Prediction Output")
    result_box = st.empty()

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

        # ---------- Display Prediction ----------
        result_box.markdown(
            f"""
            <div class='glass-card'>
                <h2 style='text-align:center;'>Predicted Digit: <b>{digit}</b> 😎</h2>
                <h4 style='text-align:center;'>Confidence: <b>{conf:.2f}%</b> ⭐</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- Live Digit Preview ----------
        st.markdown("### Processed 28×28 Image")
        st.image(pil.resize((140,140)), caption="MNIST-scaled Input", width=140)

        # ---------- TOP-3 Prediction Chart ----------
        st.markdown("### 🔝 Top-3 Probabilities")
        top_idx = prob.argsort()[-3:][::-1]
        top_values = prob[top_idx]
        top_labels = [str(i) for i in top_idx]

        fig = go.Figure(data=[
            go.Bar(x=top_labels, y=top_values, text=[f"{v*100:.2f}%" for v in top_values], textposition="outside")
        ])
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
