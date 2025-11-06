import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import cv2
import json
import os
from sklearn.model_selection import train_test_split

# Configurações específicas para seu tipo de CAPTCHA
CHARACTERS = '0123456789abcdefghijklmnopqrstuvwxyz'  # Números e letras minúsculas
CAPTCHA_LENGTH = 4
CHAR_TO_NUM = {char: i for i, char in enumerate(CHARACTERS)}
NUM_TO_CHAR = {i: char for i, char in enumerate(CHARACTERS)}

def generate_similar_captcha():
    """Gera um CAPTCHA similar aos exemplos reais"""
    width, height = 200, 50
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Usar fonte maior e mais similar
    try:
        # Tentar várias fontes até encontrar a mais similar
        fonts = ['Consolas', 'Courier', 'DejaVuSansMono']
        font = None
        for font_name in fonts:
            try:
                font = ImageFont.truetype(font_name, 42)  # Fonte bem maior
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Gerar texto
    chars = CHARACTERS[:10] * 3 + CHARACTERS[10:]
    text = ''.join(random.choice(chars) for _ in range(CAPTCHA_LENGTH))
    
    # Adicionar cada caractere separadamente com espaçamento correto
    x_start = 20
    for i, char in enumerate(text):
        # Posição x com espaçamento variável
        if i == 0:
            x = x_start
        else:
            x = x_start + (i * 45)  # Espaçamento maior entre caracteres
        
        # Adicionar caractere em cor RGB
        r = random.randint(0, 100)  # Manter cores escuras
        g = random.randint(0, 100)
        b = random.randint(0, 100)
        draw.text((x, 0), char, font=font, fill=(r,g,b))
    
    # Converter para numpy array
    img_array = np.array(image)
    
    # Adicionar linhas horizontais mais finas
    for _ in range(2):
        y = random.randint(2, 10)
        cv2.line(img_array, (0, y), (width, y), (200,200,200), 1)
    
    # Adicionar pontos coloridos
    for _ in range(2):
        y_dot = height - random.randint(15, 20)
        x_dot = random.randint(20, width-20)
        if random.random() > 0.5:
            color = (255, 0, 0)  # Vermelho
        else:
            color = (0, 0, 255)  # Azul
        cv2.circle(img_array, (x_dot, y_dot), 2, color, -1)
    
    return text, img_array

def encode_text(text):
    """Codifica o texto do CAPTCHA para one-hot encoding"""
    encoded = np.zeros((CAPTCHA_LENGTH, len(CHARACTERS)))
    for i, char in enumerate(text):
        encoded[i, CHAR_TO_NUM[char]] = 1
    return encoded.flatten()

def create_model():
    """Cria um modelo mais simples, tratando cada caractere separadamente"""
    # Entrada da imagem
    input_img = layers.Input(shape=(50, 200, 1))
    
    # Camadas convolucionais compartilhadas
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Uma saída para cada caractere
    outputs = []
    for i in range(4):  # 4 caracteres
        output = layers.Dense(36, activation='softmax', name=f'char_{i}')(x)
        outputs.append(output)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(img):
    """Pré-processamento mais robusto"""
    # Binarização adaptativa
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remover ruído
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Normalizar
    img = img.astype('float32') / 255
    return img

def encode_labels(text):
    """Codifica cada caractere separadamente"""
    labels = []
    for char in text:
        label = np.zeros(36)
        if char.isdigit():
            label[int(char)] = 1
        else:
            label[ord(char) - ord('a') + 10] = 1
        labels.append(label)
    return labels

def load_dataset():
    """Carrega dataset com novo processamento"""
    with open('dataset/labels.json', 'r') as f:
        labels = json.load(f)
    
    X = []
    y1, y2, y3, y4 = [], [], [], []
    
    print(f"Carregando {len(labels)} CAPTCHAs...")
    
    for filename, text in labels.items():
        img_path = os.path.join('dataset', filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img = cv2.resize(img, (200, 50))
        img = preprocess_image(img)
        X.append(img.reshape(50, 200, 1))
        
        # Codificar cada caractere separadamente
        char_labels = encode_labels(text)
        y1.append(char_labels[0])
        y2.append(char_labels[1])
        y3.append(char_labels[2])
        y4.append(char_labels[3])
    
    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    
    print(f"Dataset carregado: {len(X)} imagens")
    
    # Split mantendo a ordem dos caracteres
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y1_train, y1_val = y1[train_idx], y1[val_idx]
    y2_train, y2_val = y2[train_idx], y2[val_idx]
    y3_train, y3_val = y3[train_idx], y3[val_idx]
    y4_train, y4_val = y4[train_idx], y4[val_idx]
    
    train_data = (X_train, [y1_train, y2_train, y3_train, y4_train])
    val_data = (X_val, [y1_val, y2_val, y3_val, y4_val])
    
    return train_data, val_data

def augment_image(img):
    """Aplica augmentação na imagem"""
    # Ruído gaussiano
    noise = np.random.normal(0, 0.05, img.shape)
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    
    return img_noisy

def train_model():
    """Treina o modelo com dados reais"""
    model = create_model()
    
    # Carregar dados
    print("Carregando dataset...")
    (X_train, y_train), (X_val, y_val) = load_dataset()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Treinar
    print("Iniciando treinamento...")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar modelo
    print("Salvando modelo...")
    model.save('captcha_model.h5')
    
    return history

if __name__ == "__main__":
    print("Iniciando processo de treinamento...")
    history = train_model()
    print("Treinamento concluído!") 