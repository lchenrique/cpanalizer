import cv2
import json
import os
import numpy as np

def check_dataset():
    """Verifica a qualidade dos dados de treinamento"""
    with open('dataset/labels.json', 'r') as f:
        labels = json.load(f)
    
    print(f"Total de {len(labels)} CAPTCHAs rotulados")
    
    # Estatísticas dos caracteres
    char_count = {}
    for text in labels.values():
        for char in text:
            char_count[char] = char_count.get(char, 0) + 1
    
    print("\nDistribuição de caracteres:")
    for char, count in sorted(char_count.items()):
        print(f"{char}: {count}")
    
    # Verificar imagens
    for filename, text in labels.items():
        img_path = os.path.join('dataset', filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Erro na imagem: {filename}")
            continue
            
        # Mostrar imagem com texto
        cv2.imshow(f'Verificando: {text}', img)
        key = cv2.waitKey(0)
        
        if key == ord('d'):  # Pressione 'd' para deletar label incorreto
            del labels[filename]
            print(f"Removido: {filename}")
        elif key == ord('q'):
            break
    
    # Salvar labels corrigidos
    with open('dataset/labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_dataset() 