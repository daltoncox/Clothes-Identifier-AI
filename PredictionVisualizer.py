import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import NeuralNetwork as NN

import streamlit as st
import numpy as np

def best_available_device():
    device = (
        "cuda" if torch.cuda.is_available() # NVIDIA Cuda
        else "mps" if torch.backends.mps.is_available() # Apple Silicon
        else "cpu"
    )
    return device

def predict(img, model, device):
    model.eval()
    with torch.no_grad():
        prob = model(img.to(device))
        prob = torch.nn.functional.softmax(prob, dim=1)
    return prob.cpu().detach().numpy().flatten()

def next_image():
    st.session_state.img += 1
    if st.session_state.img >= len(test_data):
        st.session_state.img = 0

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

device=best_available_device()
model = NN.NeuralNetwork().to(device)
try:
    model.load_state_dict(torch.load('model/clothes_identifier.pt', weights_only=True))
except:
    st.write("Failed to load exsisting model. Please train a model first.")

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

if 'img' not in st.session_state:
    st.session_state.img = 0

img, label = test_data[st.session_state.img]
prob = predict(img, model, device)

# Format and scale image to be displayed
np_img = img.squeeze().numpy()
np_img = np.repeat(np.repeat(np_img, 10, axis=0), 10, axis=1)

st.title("Model Prediction Visualizer")
st.image(image=np_img,caption=f"{labels_map[label]}")
st.button("Next Image", on_click=next_image)
st.subheader(f"Model Prediction: {labels_map[np.argmax(prob)]}")

# Chart the probabilities
st.bar_chart(
    data=dict(zip(labels_map.values(), prob)), 
    horizontal=True, 
    x_label="Probability", 
    use_container_width=True
    )