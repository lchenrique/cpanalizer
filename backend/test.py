import requests
import os

def test_captcha(image_path):
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo não encontrado: {image_path}")
        return
    
    print(f"Testando arquivo: {image_path}")
    print(f"Caminho completo: {os.path.abspath(image_path)}")
        
    url = "http://localhost:8000/recognize-captcha"
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            print("Enviando requisição...")
            response = requests.post(url, files=files)
            print(f"Status code: {response.status_code}")
            
        result = response.json()
        print("\nResposta completa:", result)
        print("\nResultados do reconhecimento:")
        print("-" * 30)
        print("Status:", "Sucesso" if result.get("success", False) else "Falha")
        print("Texto reconhecido:", result.get("text", "N/A"))
        print("Confiança:", result.get("confidence", "N/A"))
        if "raw_text" in result:
            print("Texto bruto:", result.get("raw_text", "N/A"))
        if "error" in result:
            print("Erro:", result.get("error"))
        print("-" * 30)
        
    except Exception as e:
        print(f"Erro ao processar imagem: {str(e)}")

# Exemplo de uso
if __name__ == "__main__":
    # Nome correto do arquivo
    image_path = "captcha.png"  # Arquivo atual
    test_captcha(image_path) 