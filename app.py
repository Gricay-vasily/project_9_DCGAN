'''
Додаток на основі DCGAN для Виявлення Спроб Обходу Капчі - 2024
'''

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model

# Завантаження моделей
generator = load_model("generator.h5")
discriminator = load_model("discriminator.h5")

# Розмір латентного простору
LATENT_DIM = 100


def init_states():
    '''
    Ініціалізація початкових станів
    '''
    st.session_state.loaded_image = None
    st.session_state.generated_image = None
    
def load_image():
    '''Зававнтаження зображення'''
    l_f = st.file_uploader(label="Виберіть або перетягніть малюнок",
                                    type = ["png", "jpg", "jpeg"])
    if l_f is None:
        init_states()
    return l_f

def set_image(l_f):
    '''Встановлення стану завантаженого зображення'''
    if l_f is not None:
        # Відкриття зображення за допомогою Pillow
        st.session_state.loaded_image = Image.open(l_f)
    else:
        st.session_state.loaded_image = None

# Функція для завантаження та підготовки зображення
def prepare_image(img):
    '''
    Підготовка зображення до опрацювання його мережею
    '''
    img = img.resize((32, 32))
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)
    return img


def show_image(img, caption = "",height = 100):
    '''
    Виведення зображення з деяким його нормуванням за розмірами
    '''
    # Відкриття зображення за допомогою Pillow
    # Приведення розміру до стандартного по висоті 100
    st.write(f"Реальний розмір зображення : {img.height} x {img.width}")
    koef = height / img.height
    img = img.resize((
        int(img.width * koef), int(img.height * koef)
                      ))
    st.write(f"Зменшений розмір зображення : {img.height} x {img.width}")
    st.image(image=img,
                 caption=caption)

# ----------------------------------------------------------------------------
# Вхід в головну частину програми
if __name__ == "__main__":

    # Ініціалізація початкових станів
    if "loaded_image" not in st.session_state:
        st.session_state.loaded_image = None
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None

    # Створення веб-інтерфейсу за допомогою Streamlit
    st.title("Генератор та Дискримінатор зображень CAPTCHA")

    # Завантаження зображення
    loaded_file = load_image()

    st.button("Підтвердіть вибір", on_click=set_image(loaded_file))
    # st.button("Скинути всі зображення", on_click=init_states())

    if st.session_state.loaded_image is not None:
        show_image(img=st.session_state.loaded_image,
                   caption="Завантажене зображення",
                   height=100)

        # Підготовка зображення
        prepared_image = prepare_image(st.session_state.loaded_image)

    # Класифікація завантаженого зображення
    prediction = discriminator.predict(prepared_image)
    predicted_class = "Real" if prediction >= 0.5 else "Fake"
    probability = prediction[0][0]

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Probability: {probability:.2f}")

    # Генерація нового зображення
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    generated_img = generator.predict(noise)

    # Денормалізація згенерованого зображення
    generated_img = 0.5 * generated_img + 0.5
    generated_img = np.clip(generated_img, 0, 1)

    # Відображення згенерованого зображення
    st.session_state.generated_image = generated_img
    st.image(generated_img[0], caption="Generated Image", use_column_width=True)

    # Перевірка згенерованого зображення дискримінатором
    gen_prediction = discriminator.predict(generated_img)
    gen_predicted_class = "Real" if gen_prediction >= 0.5 else "Fake"
    gen_probability = gen_prediction[0][0]

    st.write(f"Generated Image Predicted Class: {gen_predicted_class}")
    st.write(f"Generated Image Probability: {gen_probability:.2f}")
