# Refatoração: Adaptar projeto de OCR de CAPTCHA para reconhecimento de placas brasileiras

## Contexto
Tenho um projeto funcional de reconhecimento de CAPTCHA usando TensorFlow/Keras que precisa ser adaptado para reconhecer placas de veículos brasileiras. O projeto atual:

- Reconhece CAPTCHA de 4 caracteres alfanuméricos
- Usa classificação por posição fixa (não usa CTC)
- FastAPI como backend
- Modelo treinado em `captcha_model.h5`
- Dataset rotulado em JSON

Repositório: https://github.com/lchenrique/cpanalizer

## Objetivo
Adaptar o projeto para reconhecer **placas brasileiras** nos dois formatos oficiais:

1. **Formato Antigo**: `AAA-0000` (3 letras + 4 números)
2. **Formato Mercosul**: `AAA0A00` (3 letras + 1 número + 1 letra + 2 números)

Ambos têm **7 caracteres** (desconsiderando hífen).

## Mudanças necessárias

### 1. Arquitetura do modelo
- [ ] Alterar de **4 posições** para **7 posições** fixas
- [ ] Ajustar charset para apenas letras maiúsculas e números: `CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'` (36 caracteres)
- [ ] Manter a mesma arquitetura de classificação por posição (não precisa CTC)
- [ ] Atualizar input shape se necessário (avaliar melhor dimensão para placas)

### 2. Pré-processamento
- [ ] Ajustar `process_image_efficiently()` para aceitar imagens de placas (podem ter proporções diferentes de CAPTCHA)
- [ ] Adicionar normalização/binarização específica para placas (reflexos, iluminação variável)
- [ ] Considerar dois tipos visuais: placa antiga (cinza) e Mercosul (branca com QR code)

### 3. Pós-processamento e validação
- [ ] Implementar função `validar_placa_brasileira()` que:
  - Identifica formato (antiga vs Mercosul) por regex
  - Valida estrutura: antiga `[A-Z]{3}[0-9]{4}` ou Mercosul `[A-Z]{3}[0-9][A-Z][0-9]{2}`
  - Retorna placa formatada + tipo + status de validade

- [ ] Implementar correção automática de confusões por posição:
  - **Formato antigo**: posições 0-2 = só letras, posições 3-6 = só números
  - **Formato Mercosul**: pos 0-2 = letras, pos 3 = número, pos 4 = letra, pos 5-6 = números
  - Aplicar mapa de confusões: `O↔0`, `I↔1`, `B↔8`, `S↔5`, `Z↔2`, `G↔6`

### 4. Endpoints da API
- [ ] Criar endpoint `/recognize-plate` para reconhecer placas
- [ ] Retornar JSON com:
  ```json
  {
    "success": true,
    "placa": "ABC-1234",
    "formato": "antiga",
    "valida": true,
    "confianca": 0.95,
    "texto_bruto": "ABC1234"
  }
  ```
- [ ] Manter endpoint antigo `/recognize-captcha` para retrocompatibilidade (opcional)

### 5. Dataset e treinamento
- [ ] Atualizar `train_model.py`:
  - Mudar `CAPTCHA_LENGTH = 4` para `CAPTCHA_LENGTH = 7`
  - Atualizar charset
  - Ajustar função de geração sintética para criar placas (ou remover se for usar dataset real)
  
- [ ] Criar estrutura de dataset:
  ```
dataset/
  ├── antiga/
  │   ├── placa_001.png
  │   └── ...
  ├── mercosul/
  │   ├── placa_001.png
  │   └── ...
  └── labels.json
  ```

- [ ] Atualizar `labels.json` para formato de 7 caracteres:
  ```json
  {
    "antiga/placa_001.png": "ABC1234",
    "mercosul/placa_001.png": "ABC1D23"
  }
  ```

### 6. Data augmentation específico para placas
- [ ] Adicionar transformações realistas:
  - Perspective transform (ângulos de câmera)
  - Motion blur (carro em movimento)
  - Reflexo/brilho
  - Sujeira/desgaste
  - Variação de iluminação (dia/noite)
  - Compressão JPEG

### 7. Detecção de placa (YOLO) - **FASE 2** (opcional para primeira versão)
- [ ] Integrar YOLOv8 para detectar bounding box da placa na imagem do veículo
- [ ] Pipeline completo: `imagem → YOLO detecção → crop → OCR → texto`
- [ ] Criar endpoint `/detect-and-recognize` que aceita foto completa do carro

### 8. Configurações e documentação
- [ ] Atualizar `README.md` com:
  - Descrição do projeto de reconhecimento de placas
  - Formatos suportados (antiga e Mercosul)
  - Instruções de uso
  - Exemplos de requisição/resposta

- [ ] Atualizar Docker:
  - Renomear modelo de `captcha_model.h5` para `plate_model.h5`
  - Ajustar variáveis de ambiente se necessário

- [ ] Criar arquivo `config.py` para centralizar constantes:
  ```python
  PLATE_LENGTH = 7
  CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  IMAGE_WIDTH = 200  # Ajustar conforme necessário
  IMAGE_HEIGHT = 50
  ```

### 9. Testes e validação
- [ ] Criar testes unitários para:
  - Validação de formato
  - Correção de confusões
  - Pré-processamento
  
- [ ] Testar com imagens reais de placas antigas e Mercosul
- [ ] Validar accuracy em dataset separado (test set)

## Arquivos principais a modificar

1. **`main.py`**
   - Função `decode_prediction()` → 7 posições
   - Adicionar `validar_placa_brasileira()`
   - Adicionar `corrigir_confusoes_placa()`
   - Novo endpoint `/recognize-plate`

2. **`train_model.py`**
   - Constante `CAPTCHA_LENGTH = 7`
   - Atualizar `CHARACTERS`
   - Ajustar função `encode_text()` para 7 caracteres

3. **`solve_captcha.py`** → renomear para `solve_plate.py`
   - Atualizar `decode_predictions()` para 7 posições
   - Ajustar charset

4. **`collect_captchas.py`** → adaptar ou criar `collect_plates.py`
   - Script para coletar/preparar dataset de placas

5. **`label_captchas.py`** → adaptar para `label_plates.py`
   - Interface para rotular placas (7 caracteres)

## Restrições e considerações

- ✅ Manter arquitetura de classificação por posição fixa (não migrar para CTC nesta fase)
- ✅ Suportar ambos formatos (antiga e Mercosul) com um único modelo
- ✅ Foco inicial em imagens de placas **já recortadas** (detector YOLO fica para fase 2)
- ✅ Charset brasileiro: 26 letras (A-Z) + 10 dígitos (0-9) = 36 classes
- ⚠️ Considerar que placas podem ter reflexo, sujeira, ângulos variados
- ⚠️ Mercosul tem QR code no canto (não deve interferir se crop for correto)

## Resultado esperado

Ao final da refatoração, o projeto deve:

1. ✅ Aceitar imagem de placa recortada via API
2. ✅ Retornar texto da placa + formato + validação
3. ✅ Corrigir automaticamente confusões baseado em posição
4. ✅ Funcionar para placas antigas e Mercosul
5. ✅ Ser retreinável com novo dataset de placas
6. ✅ Manter estrutura FastAPI + Docker funcionando

## Prioridade de implementação

**Fase 1 (MVP):**
1. Ajustar modelo para 7 posições
2. Atualizar endpoints e validação
3. Retreinar com dataset mínimo (mesmo que sintético)
4. Testar com imagens recortadas

**Fase 2 (Produção):**
5. Adicionar correção de confusões
6. Integrar YOLO para detecção automática
7. Melhorar data augmentation
8. Otimizar accuracy com dataset real robusto

---

## Como usar este documento

Este documento serve como:
- ✅ **Checklist** para refatoração manual
- ✅ **Prompt** para assistentes de IA (Copilot, ChatGPT, Claude)
- ✅ **Especificação** para abertura de issues/PRs
- ✅ **Documentação** do processo de migração

---

**Status**: 📋 Planejamento
**Versão**: 1.0
**Data**: 2026-02-12