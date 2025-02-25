import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# モデル定義
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden_activation)
        logvar = self.fc_logvar(hidden_activation)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_latent))
        output = torch.sigmoid(self.fc_out(hidden_activation))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

# モデルロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(latent_dim=3, num_classes=10).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("CVAE 数字画像生成")
st.sidebar.header("入力")
label = st.sidebar.selectbox("生成する数字", list(range(10)), index=5)
n_samples = st.sidebar.slider("生成枚数", min_value=1, max_value=10, value=6)

if st.sidebar.button("画像生成"):
    with torch.no_grad():
        z = torch.randn(n_samples, 3).to(device)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=device)
        x_gen = model.decoder(z, labels)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(x_gen[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
