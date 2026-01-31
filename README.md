# EdgeVision RAG: Pesquisa Semântica em Vídeo (Edge AI + GenAI)

Este projeto implementa uma pipeline completa de **Visão Computacional na Borda (Edge AI)** integrada com **Recuperação de Informação Multimodal (RAG)**. 

O sistema é capaz de "assistir" a um vídeo, compreender o conteúdo semântico (quem está fazendo o que), e permitir que humanos façam buscas textuais complexas para encontrar momentos específicos, como *"pessoa andando de roupa vermelha"* ou *"alguém correndo"*.

## 🏗️ Arquitetura da Solução

O projeto foi desenhado para **alta performance em CPU** e compatibilidade com ambientes modernos (Python 3.14+).

### 1. Ingestão e Processamento (Pipeline)
*   **Entrada**: Vídeo MP4 ou Webinar (Webcam).
*   **Detecção de Objetos (Edge AI)**: Utiliza **YOLOv8 Nano** convertido para **ONNX**. 
    *   *Diferencial*: Execução via `OpenCV DNN Module` para evitar dependências pesadas de GPU e garantir compatibilidade.
    *   *Otimização*: Redimensionamento de frames (640x480) e "Frame Skipping" (processa embeddings a cada 1s) para rodar liso em CPUs convencionais.
*   **Extração de Características (Feature Extraction)**: Recortes (crops) das pessoas detectadas são enviados para o modelo **CLIP (ViT-B-32)**.
    *   O CLIP converte a imagem em um vetor de 512 dimensões que representa o "significado" visual.

### 2. Armazenamento Vetorial (Vector DB)
*   **Estratégia Híbrida**: 
    1.  Tenta utilizar **ChromaDB** para armazenamento persistente.
    2.  **Fallback Automático**: Se o ambiente não suportar HNSW (comum em Python muito novo ou Windows ARM), o sistema degrada graciosamente para **Scikit-Learn (KNN) + Pickle**, mantendo 100% da funcionalidade.
*   **Gerenciamento de Ciclo de Vida**: Suporte nativo a **Reset** (limpeza de banco) ou **Acúmulo** (indexar múltiplos vídeos).

### 3. Backend e API
*   **FastAPI**: Servidor assíncrono de alta performance.
*   **Busca Semântica**: Endpoint `/search` converte texto em vetor (via CLIP Text Encoder) e busca vizinhos mais próximos (Cosine Similarity) no banco de imagens.

### 4. Observabilidade (Monitoramento)
*   **Instrumentação**: A API expõe métricas via protocolo **Prometheus** (`/metrics`).
*   **Dashboard Visual**: Stack **Docker** com **Prometheus** (coleta) e **Grafana** (visualização) pré-configurados em Português.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
*   **Python 3.9+** (Testado e validado no Python 3.14)
*   **Docker & Docker Compose** (Apenas para o painel de monitoramento Grafana)

### Passo 1: Instalação e Configuração

1.  **Clone e configure o ambiente**:
    ```powershell
    # Criação do ambiente virtual
    python -m venv venv
    .\venv\Scripts\Activate
    
    # Instalação das dependências
    pip install -r requirements.txt
    ```

2.  **Prepare os Modelos e Dados**:
    ```powershell
    # Baixa vídeo de exemplo
    python scripts/download_data.py
    
    # Exporta o modelo YOLO para formato otimizado ONNX
    python scripts/export_model.py
    ```

### Passo 2: Ingestão de Vídeo (Pipeline)

Execute o processamento do vídeo. Você tem dois modos de operação:

*   **Modo Padrão (Acumulativo)**: Mantém os dados antigos e adiciona os novos. Ideal para indexar uma biblioteca de vídeos.
    ```powershell
    python run_pipeline.py "data/people-walking.mp4"
    ```

*   **Modo Reset (Limpeza)**: Apaga todo o banco de dados antes de começar. Ideal para testes ou reprocessamento limpo.
    ```powershell
    python run_pipeline.py "data/people-walking.mp4" --reset
    ```

    *Dica: Pressione `q` na janela de vídeo para interromper o processo antecipadamente.*

### Passo 3: Iniciar a API de Busca

Com o banco de dados populado, inicie o servidor:

```powershell
python run_api.py
```
> O servidor iniciará em `http://localhost:8000`.

*   **Documentação Interativa (Swagger)**: Acesso em [http://localhost:8000/docs](http://localhost:8000/docs).
*   **Teste Rápido**: Use o endpoint `/search` para buscar por termos como "person", "red shirt", "walking".

### Passo 4: Monitoramento (Dashboard) 📊

Para visualizar métricas de performance (RPS, Total de Buscas) em tempo real:

1.  **Suba a Stack de Observabilidade**:
    ```powershell
    docker-compose up -d
    ```

2.  **Acesse o Grafana**:
    *   URL: [http://localhost:3000](http://localhost:3000)
    *   Login: `admin` / `admin`
    *   Vá em **Dashboards > Default > Estatísticas da API EdgeVision**.

---

## 🛠️ Ferramentas de Verificação

O projeto inclui scripts utilitários na pasta `scripts/` para diagnóstico:

*   **`python scripts/check_db.py`**: Verifica integridade do banco de dados local (Pickle) e conta quantos vetores existem.
*   **`python scripts/verify_api.py`**: Realiza um teste "End-to-End". Conecta na API, verifica status e faz uma busca teste, exibindo tempo de latência e distância de similaridade.

---

## 📂 Estrutura de Arquivos

```text
EdgeVision RAG/
├── data/                   # Vídeos e Banco de Dados (Pickle/Chroma)
├── grafana/                # Configurações de Provisioning (IaC) e Dashboards
├── scripts/                # Scripts auxiliares (download, export, verify)
├── src/
│   ├── api.py              # Aplicação FastAPI
│   ├── feature_extractor.py# Wrapper do Modelo CLIP
│   ├── inference_engine.py # Wrapper do YOLO/ONNX
│   └── vector_db.py        # Wrapper do Banco Vetorial (com Fallback)
├── docker-compose.yml      # Orquestração Grafana + Prometheus
├── run_pipeline.py         # Script Principal de Ingestão
├── run_api.py              # Entrypoint da API
└── requirements.txt        # Dependências Python
```

---

*Desenvolvido em 2026 como referência técnica de Arquitetura de Software para IA.*
