import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

def preprocess_image(img):
    """Pré-processamento da imagem"""
    # Binarização adaptativa
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remover ruído
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Normalizar
    img = img.astype('float32') / 255
    return img

def decode_predictions(predictions):
    """Decodifica as previsões do modelo"""
    CHARACTERS = '0123456789abcdefghijklmnopqrstuvwxyz'
    text = ''
    
    for pred in predictions:
        char_idx = np.argmax(pred)
        if char_idx < 10:  # Número
            text += str(char_idx)
        else:  # Letra
            text += CHARACTERS[char_idx]
    
    return text

def solve_captcha(img_bytes):
    """Resolve um CAPTCHA a partir dos bytes da imagem"""
    # Carregar modelo
    model = tf.keras.models.load_model('captcha_model.h5')
    
    # Converter bytes para imagem
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Pré-processar
    img = cv2.resize(img, (200, 50))
    img = preprocess_image(img)
    img = img.reshape(1, 50, 200, 1)
    
    # Predição
    predictions = model.predict(img, verbose=0)
    return decode_predictions([p[0] for p in predictions]) 