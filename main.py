import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import CVAE  # モデル定義ファイルを別途作成

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim=3, num_classes=10).to(device)
    model.load_state_dict(torch.load("cvae.pth", map_location=device))
    model.eval()
    return model, device

def generate_digit(model, device, digit, n_samples=6):
    latent_dim = 3
    z = torch.randn(n_samples, latent_dim).to(device)
    y = torch.full((n_samples,), digit, dtype=torch.long, device=device)
    with torch.no_grad():
        x_gen = model.decoder(z, y)
    return x_gen.cpu().numpy()

st.title("Conditional Variational Autoencoder (CVAE) Demo")
model, device = load_model()

digit = st.slider("Choose a digit to generate", 0, 9, 5)
if st.button("Generate"):
    images = generate_digit(model, device, digit)
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
