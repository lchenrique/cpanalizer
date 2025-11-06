import os
import cv2
import json
from datetime import datetime

def label_captchas(quantidade=50):
    """
    Interface simples para rotular CAPTCHAs
    """
    labels_file = 'dataset/labels.json'
    labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    
    dataset_dir = 'dataset'
    captcha_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png') and f not in labels]
    
    print(f"Vamos rotular {quantidade} CAPTCHAs")
    
    try:
        for i, filename in enumerate(captcha_files[:quantidade]):
            # Mostrar imagem
            img_path = os.path.join(dataset_dir, filename)
            img = cv2.imread(img_path)
            
            # Mostrar em tamanho maior
            img = cv2.resize(img, (400, 100))  # 2x maior
            cv2.imshow('CAPTCHA', img)
            cv2.waitKey(100)
            
            # Pedir input
            text = input(f"CAPTCHA {i+1}/{quantidade} - Digite o texto (ou 'q' para sair): ").lower()
            
            if text == 'q':
                break
            
            if len(text) == 4 and text.isalnum():
                labels[filename] = text
                with open(labels_file, 'w') as f:
                    json.dump(labels, f, indent=2)
                print(f"✓ Salvo: {text} ({i+1}/{quantidade})")
            else:
                print("❌ Texto inválido! Digite 4 caracteres alfanuméricos.")
                
    except KeyboardInterrupt:
        print("\nProcesso interrompido")
    finally:
        cv2.destroyAllWindows()
        
    print(f"\nTotal rotulado: {len(labels)}")
    return labels

if __name__ == "__main__":
    label_captchas(50)  # Começar com 50 CAPTCHAs 