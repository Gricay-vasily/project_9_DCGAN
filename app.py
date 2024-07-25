import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import subprocess

# Завантаження моделей
generator = load_model("generator.h5")
discriminator = load_model("discriminator.h5")

# Розмір латентного простору
latent_dim = 100


# Функція для завантаження та підготовки зображення
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)
    return img


# Створення веб-інтерфейсу за допомогою Streamlit
st.title("CAPTCHA Classifier and Generator")

# Завантаження зображення
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Відображення завантаженого зображення
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Підготовка зображення
    img = load_image(uploaded_file)

    # Класифікація завантаженого зображення
    prediction = discriminator.predict(img)
    predicted_class = "Real" if prediction >= 0.5 else "Fake"
    probability = prediction[0][0]

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Probability: {probability:.2f}")

    # Генерація нового зображення
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_img = generator.predict(noise)

    # Денормалізація згенерованого зображення
    generated_img = 0.5 * generated_img + 0.5
    generated_img = np.clip(generated_img, 0, 1)

    # Відображення згенерованого зображення
    st.image(generated_img[0], caption="Generated Image", use_column_width=True)

    # Перевірка згенерованого зображення дискримінатором
    gen_prediction = discriminator.predict(generated_img)
    gen_predicted_class = "Real" if gen_prediction >= 0.5 else "Fake"
    gen_probability = gen_prediction[0][0]

    st.write(f"Generated Image Predicted Class: {gen_predicted_class}")
    st.write(f"Generated Image Probability: {gen_probability:.2f}")
    
if __name__ == "main":
    # os.system("streamlit run app.py")
    # subprocess.run(["python", "-m", "streamlit", "run"], check=False)
