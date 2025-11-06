import requests
import os
import time
from datetime import datetime

def download_captcha(session, index):
    """
    Baixa um CAPTCHA do site e salva com timestamp
    """
    try:
        # URL do CAPTCHA
        url = "http://www.proeisbm.cbmerj.rj.gov.br/captcha2.php"
        
        # Fazer requisição
        response = session.get(url)
        
        if response.status_code == 200:
            # Criar pasta se não existir
            os.makedirs('dataset', exist_ok=True)
            
            # Salvar imagem com timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'dataset/captcha_{timestamp}_{index}.png'
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Baixado CAPTCHA {index}")
            return True
    except Exception as e:
        print(f"Erro ao baixar CAPTCHA {index}: {str(e)}")
    
    return False

def collect_captchas(quantidade=200):
    """
    Coleta uma quantidade específica de CAPTCHAs
    """
    session = requests.Session()
    
    print(f"Iniciando coleta de {quantidade} CAPTCHAs...")
    
    for i in range(quantidade):
        success = download_captcha(session, i)
        
        if success:
            time.sleep(1.5)
        else:
            print(f"Falha ao baixar CAPTCHA {i}")
        
        if (i + 1) % 10 == 0:
            print(f"Progresso: {i + 1}/{quantidade}")

if __name__ == "__main__":
    # Coletar 200 CAPTCHAs
    collect_captchas(200) 