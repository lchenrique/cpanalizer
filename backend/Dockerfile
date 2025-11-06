FROM python:3.10-slim

# Configurar variáveis de ambiente não interativo
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Instalar dependências do sistema e limpar cache em uma única camada
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar apenas requirements primeiro para aproveitar cache
COPY requirements.txt .

# Instalar dependências Python com pip em uma única camada
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Criar usuário não-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copiar o modelo primeiro (arquivo grande)
COPY --chown=appuser:appuser captcha_model.h5 .

# Verificar se o modelo existe
RUN if [ ! -f captcha_model.h5 ]; then echo "Erro: captcha_model.h5 não encontrado" && exit 1; fi

# Copiar o resto dos arquivos do projeto
COPY --chown=appuser:appuser . .

# Mudar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "debug"]