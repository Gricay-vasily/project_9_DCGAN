'''
Додаток на основі DCGAN для Виявлення Спроб Обходу Капчі - 2024
'''

from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from settings import IMAGE_MAX_HEIGTH,\
                        MODELS_DIR, GEN_FILE, DISC_FILE

# Завантаження моделей
def load_models():
    '''Завантаження моделей'''
    gen_file = Path(MODELS_DIR) / Path(GEN_FILE)
    disc_file = Path(MODELS_DIR) / Path(DISC_FILE)
    gen = load_model(gen_file)
    disc = load_model(disc_file)
    return gen, disc

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
    init_states()
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

def gen_show_image(gen, disc, latent_dim):
    '''Генерація зображення'''
    # st.write(f"real_img.shape = {real_img.shape}")
    noise = tf.random.normal([1, latent_dim])
    # st.write(f"noise shape = {noise.shape}")
    generated_img = gen.predict(noise, verbose=0)
    print(np.info(generated_img))
    # Відображення згенерованого зображення
    st.session_state.generated_image = Image.fromarray(generated_img[0], "RGB")
    show_image(img=st.session_state.generated_image,
                caption="Згенероване зображення",
                height=32)
    show_predictions(disc=disc,
                     img=generated_img,
                     img_class="згенерованого")

def show_image(img, caption = "",height = 100):
    '''
    Виведення зображення з деяким його нормуванням за розмірами
    '''
    # Приведення розміру до стандартного по висоті 100
    st.write(f"Реальний розмір зображення : {img.height} x {img.width}")
    koef = height / img.height
    img = img.resize((
        int(img.width * koef), int(img.height * koef)
                      ))
    st.write(f"Перетворений розмір зображення : {img.height} x {img.width}")
    st.image(image=img,
                 caption=caption)

def show_predictions(disc, img, img_class:str):
    '''Виведення результатів роботи дискримінатора'''
    prediction = disc.predict(img, verbose=0)
    # prediction = 0.3
    predicted_class = "Real" if prediction >= 0.5 else "Fake"
    probability = prediction[0][0]
    st.write(f"Передбачений клас для {img_class} зображення: {predicted_class}")
    st.write(f"Імовірність вірної класифікації для {img_class} зображення: {probability:.2f}")



# ----------------------------------------------------------------------------
# Вхід в головну частину програми
if __name__ == "__main__":
    # Завантаження моделей
    generator, discriminator = load_models()

    # print(f"GENERTATOR\n{generator.summary()}")
    # print(f"DISCRIMINATOR\n{discriminator.summary()}")

    # Ініціалізація початкових станів
    if "loaded_image" not in st.session_state:
        st.session_state.loaded_image = None
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None

    # Створення веб-інтерфейсу за допомогою Streamlit
    st.title("Генератор та Дискримінатор зображень CAPTCHA")

    # Завантаження зображення
    loaded_file = load_image()

    st.button("Підтвердіть вибір",
              key="Real_Img")
    if st.button:
        set_image(loaded_file)

    if st.session_state.loaded_image is not None:
        show_image(img=st.session_state.loaded_image,
                   caption="Завантажене зображення",
                   height=IMAGE_MAX_HEIGTH)

        # Підготовка зображення
        prepared_image = prepare_image(st.session_state.loaded_image)

        # Класифікація завантаженого зображення
        show_predictions(disc=discriminator, img=prepared_image, img_class="справжнього")

        # Генерація нового зображення
        st.title("Генерація зображення")
        # st.write(f"Prepare IMG shape = {prepared_image.shape}")
        st.button("Згенерувати зображення",
                  key="Gen_Img",
                  on_click=gen_show_image(generator, discriminator, LATENT_DIM)
                  )
