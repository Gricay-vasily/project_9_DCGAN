import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Попередньо збудовані моделі генератора і дискримінатора
generator = build_generator()
discriminator = build_discriminator()

# Завантаження збережених вагів (замініть шлях на фактичний)
generator.load_weights('generator_weights.h5')
discriminator.load_weights('discriminator_weights.h5')

def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, [32, 32])
    image = (image - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return image

def main():
    st.title("CAPTCHA GAN: Генерація та Ідентифікація Зображень")
    
    uploaded_file = st.file_uploader("Завантажте зображення CAPTCHA", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Завантажене зображення', use_column_width=True)
        
        processed_image = preprocess_image(image)
        
        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        
        real_output = discriminator(tf.expand_dims(processed_image, axis=0), training=False)
        fake_output = discriminator(generated_image, training=False)
        
        verdict = "Справжнє" if real_output > fake_output else "Підроблене"
        
        st.write(f"Вердикт: {verdict}")
        
        st.image((generated_image[0] * 127.5 + 127.5).numpy().astype(np.uint8), caption='Згенероване зображення', use_column_width=True)

if __name__ == '__main__':
    main()
