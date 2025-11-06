import requests
import os

def download_test_captcha():
    """Baixa um CAPTCHA para teste"""
    url = "http://www.proeisbm.cbmerj.rj.gov.br/captcha2.php"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open('captcha.png', 'wb') as f:
            f.write(response.content)
        print("CAPTCHA baixado como 'captcha.png'")
    else:
        print("Erro ao baixar CAPTCHA")

if __name__ == "__main__":
    download_test_captcha() 