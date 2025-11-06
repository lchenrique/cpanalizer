import tensorflow as tf
import cv2
import numpy as np
import json
import os

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

def test_model():
    """Testa o modelo em CAPTCHAs reais"""
    # Carregar modelo
    model = tf.keras.models.load_model('captcha_model.h5')
    
    # Carregar labels para comparação
    with open('dataset/labels.json', 'r') as f:
        labels = json.load(f)
    
    total = 0
    correct = 0
    char_correct = [0, 0, 0, 0]
    
    print("\nTestando modelo...")
    
    for filename, real_text in labels.items():
        # Carregar e processar imagem
        img_path = os.path.join('dataset', filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 50))
        img = preprocess_image(img)
        img = img.reshape(1, 50, 200, 1)
        
        # Fazer predição
        predictions = model.predict(img, verbose=0)
        predicted_text = decode_predictions([p[0] for p in predictions])
        
        # Verificar acertos
        total += 1
        is_correct = predicted_text == real_text
        if is_correct:
            correct += 1
            
        # Verificar acertos por caractere
        for i, (p, r) in enumerate(zip(predicted_text, real_text)):
            if p == r:
                char_correct[i] += 1
        
        # Mostrar resultado
        status = "✓" if is_correct else "✗"
        print(f"{status} Real: {real_text} | Previsto: {predicted_text}")
        
        # Mostrar imagem
        img_show = cv2.imread(img_path)
        cv2.imshow('CAPTCHA', img_show)
        key = cv2.waitKey(500)  # Espera 0.5 segundos
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Mostrar estatísticas
    print("\nResultados:")
    print(f"Total testado: {total}")
    print(f"Acertos totais: {correct} ({(correct/total)*100:.2f}%)")
    print("\nAcurácia por posição:")
    for i, acc in enumerate(char_correct):
        print(f"Caractere {i+1}: {(acc/total)*100:.2f}%")

if __name__ == "__main__":
    test_model() 