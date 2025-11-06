from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import os
from solve_captcha import solve_captcha

app = FastAPI()

# Variável global para o modelo
model = None

def load_model():
    global model
    try:
        if not os.path.exists('captcha_model.h5'):
            raise FileNotFoundError("Arquivo do modelo não encontrado")
        model = tf.keras.models.load_model('captcha_model.h5')
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        model = None

# Carregar modelo na inicialização
load_model()

# Configurações
CHARACTERS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_NUM = {char: i for i, char in enumerate(CHARACTERS)}
NUM_TO_CHAR = {i: char for i, char in enumerate(CHARACTERS)}

def process_image_efficiently(file_content):
    """
    Processa a imagem de forma eficiente para o modelo
    """
    image = Image.open(io.BytesIO(file_content))
    img_np = np.array(image)
    
    # Converter para escala de cinza
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Redimensionar para o tamanho esperado pelo modelo
    resized = cv2.resize(gray, (200, 50))
    
    # Normalizar
    normalized = resized / 255.0
    
    # Preparar para o modelo
    return normalized.reshape(1, 50, 200, 1)

def decode_prediction(pred):
    """
    Decodifica a previsão do modelo para texto
    """
    output = np.reshape(pred, (4, len(CHARACTERS)))
    text = ''
    for i in range(4):
        char_idx = np.argmax(output[i])
        text += NUM_TO_CHAR[char_idx]
    return text

@app.post("/recognize-captcha")
async def recognize_captcha(file: UploadFile = File(...)):
    if model is None:
        return {
            "success": False,
            "error": "Modelo não está carregado. Verifique os logs do servidor."
        }
    
    try:
        # Ler e processar a imagem
        contents = await file.read()
        processed_image = process_image_efficiently(contents)
        
        # Fazer a previsão
        prediction = model.predict(processed_image)
        text = decode_prediction(prediction)
        
        return {
            "success": True,
            "text": text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/solve_captcha")
async def solve_captcha_endpoint(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
            
        contents = await file.read()
        text = solve_captcha(contents)
        return {"success": True, "text": text}
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 