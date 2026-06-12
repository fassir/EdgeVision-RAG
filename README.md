<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1F9BD4,50:2E75B6,100:16265F&height=210&section=header&text=EdgeVision-RAG&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Visão+Computacional+·+Embeddings+Semânticos+·+Busca+por+Texto+Natural&descAlignY=58&descSize=17&animation=fadeIn" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-1F9BD4?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2E75B6?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-16265F?style=for-the-badge)](https://ultralytics.com)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B--32-1F9BD4?style=for-the-badge)](https://openai.com/research/clip)
[![Status](https://img.shields.io/badge/Status-Ativo-2E75B6?style=for-the-badge)](https://github.com/fassir/EdgeVision-RAG)

</div>

---

## 🔭 Sobre o Projeto

<div align="center">

> *"Imagine buscar 'pessoa andando de roupa vermelha' em horas de vídeo e encontrar exatamente isso em milissegundos. Isso é EdgeVision-RAG."*

</div>

O **EdgeVision-RAG** é um pipeline avançado que combina **visão computacional** e **busca semântica RAG** para permitir consultas em linguagem natural sobre conteúdo visual. Você pode descrever o que procura em texto — `"pessoa andando de roupa vermelha"` — e o sistema encontra os frames correspondentes usando embeddings visuais e busca vetorial.

O projeto foi desenhado para rodar em **hardware limitado (edge)**: YOLOv8 Nano em ONNX via OpenCV DNN (sem GPU), CLIP para embeddings multimodais, e ChromaDB como vetor store principal com fallback inteligente para Scikit-Learn KNN.

### 🎯 Casos de Uso

| Caso | Consulta de exemplo | Resultado |
|---|---|---|
| 🔍 **Vigilância** | "pessoa com mochila vermelha" | Frames com alerta |
| 🏪 **Varejo** | "cliente na área de eletrônicos" | Análise de tráfego |
| 🎬 **Indexação de vídeo** | "cena de reunião em sala branca" | Timestamps |
| 🚗 **Trânsito** | "veículo azul na faixa da direita" | Logs filtrados |
| 🏋️ **Esportes** | "atleta chutando bola" | Análise de jogo |

---

## 🏗️ Arquitetura Completa

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE EDGEVISION-RAG                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ① INGESTÃO              ② DETECÇÃO            ③ EMBEDDING            │
│                                                                          │
│   📹 Vídeo/Stream   ──►  YOLOv8 Nano    ──►   CLIP ViT-B-32           │
│   🖼️  Imagens              ONNX via               Embedding 512-D       │
│                            OpenCV DNN             por frame/objeto       │
│                            (sem GPU)                                     │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ④ VECTOR DB             ⑤ API                 ⑥ DASHBOARD            │
│                                                                          │
│   ChromaDB         ──►   FastAPI         ──►   Prometheus               │
│   (principal)             POST /search           + Grafana               │
│   │                       query em texto         (Docker)                │
│   └─► Fallback:           retorna frames                                 │
│       sklearn KNN         similares                                      │
│       + Pickle                                                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Fluxo de Dados Detalhado

```
INPUT: frame.jpg ou stream de vídeo
       │
       ▼
[YOLOv8 Nano ONNX]          ← Detecção de objetos sem GPU
       │                        Modelo .onnx via OpenCV DNN
       ▼
[Crops de objetos detectados]   ← pessoa, veículo, animal...
       │
       ▼
[CLIP ViT-B-32]              ← Embedding visual 512 dimensões
       │                        Alinhado com espaço textual
       ▼
[Vector Store]               ← ChromaDB (primary)
       │                        sklearn KNN + Pickle (fallback)
       │
   ┌───┴──────────────────────────────────────┐
   │              BUSCA                        │
   │                                           │
   │  POST /search                             │
   │  {"query": "pessoa de roupa vermelha"}    │
   │         │                                 │
   │         ▼                                 │
   │  [CLIP text encoder]  → embedding 512-D  │
   │         │                                 │
   │         ▼                                 │
   │  [Vector similarity]  → top-K frames     │
   └───────────────────────────────────────────┘
       │
       ▼
   Resposta JSON com frames, scores e metadados
```

---

## 🛠️ Stack de Tecnologias

<div align="center">

[![My Skills](https://skillicons.dev/icons?i=python,fastapi,docker,opencv,grafana&theme=dark)](https://skillicons.dev)

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-2E75B6?style=flat-square&logo=databricks&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=grafana&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

| Componente | Tecnologia | Versão | Função |
|---|---|---|---|
| **Detecção** | YOLOv8 Nano ONNX | v8 | Detecção de objetos sem GPU |
| **Inferência** | OpenCV DNN | ≥ 4.7 | Runtime ONNX em CPU |
| **Embeddings** | CLIP ViT-B-32 | — | Vetores 512-D texto+imagem |
| **Vector DB** | ChromaDB | ≥ 0.4 | Armazenamento e busca vetorial |
| **Fallback DB** | Scikit-Learn KNN + Pickle | ≥ 1.3 | Backup sem dependências externas |
| **API** | FastAPI + Uvicorn | ≥ 0.100 | Endpoint de busca semântica |
| **Monitoramento** | Prometheus | ≥ 2.45 | Métricas de latência e volume |
| **Dashboard** | Grafana | ≥ 10.0 | Visualização das métricas |
| **Infraestrutura** | Docker Compose | ≥ 2.20 | Orquestração dos serviços |

---

## 🚀 Instalação e Execução

<details>
<summary><b>📦 1. Pré-requisitos</b></summary>

- Python 3.10+
- Docker e Docker Compose instalados
- ~4 GB de RAM disponível
- CPU moderna (sem necessidade de GPU)

```bash
# Verifique o Docker
docker --version       # Docker 20.x+
docker compose version # Compose 2.x+
```

</details>

<details>
<summary><b>⬇️ 2. Clone o repositório</b></summary>

```bash
git clone https://github.com/fassir/EdgeVision-RAG.git
cd EdgeVision-RAG
```

</details>

<details>
<summary><b>🔧 3. Instale dependências Python</b></summary>

```bash
# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Instale todas as dependências
pip install -r requirements.txt
```

```
# requirements.txt resumido:
opencv-python>=4.7
ultralytics>=8.0          # YOLOv8 (para exportar ONNX)
open-clip-torch>=2.20     # CLIP ViT-B-32
chromadb>=0.4
scikit-learn>=1.3
fastapi>=0.100
uvicorn>=0.23
prometheus-client>=0.17
numpy>=1.24
Pillow>=9.0
```

</details>

<details>
<summary><b>🐳 4. Inicie monitoramento (Docker)</b></summary>

```bash
# Sobe Prometheus + Grafana
docker compose up -d

# Acesse o Grafana em: http://localhost:3000
# Usuário: admin | Senha: admin
```

```yaml
# docker-compose.yml (resumo)
services:
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

</details>

<details>
<summary><b>🔄 5. Execute o pipeline de ingestão</b></summary>

```python
# Ingerir um vídeo ou pasta de imagens
python ingest.py --source ./videos/meu_video.mp4

# Ou ingerir imagens de uma pasta
python ingest.py --source ./frames/ --batch 32
```

```
# Saída esperada:
# [INFO] Carregando YOLOv8 Nano ONNX...
# [INFO] Carregando CLIP ViT-B-32...
# [INFO] Conectando ao ChromaDB...
# [PROGRESS] Processando frame 0001/2400 | FPS: 12.3
# [PROGRESS] Processando frame 0100/2400 | FPS: 11.8
# ...
# [INFO] Ingestão concluída: 2400 frames | 847 objetos indexados
```

</details>

<details>
<summary><b>▶️ 6. Inicie a API de busca</b></summary>

```bash
# Inicia a FastAPI
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# API disponível em: http://localhost:8000
# Docs interativas: http://localhost:8000/docs
```

```bash
# Busca por texto via curl
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "pessoa andando de roupa vermelha", "top_k": 5}'
```

```json
// Resposta da API
{
  "query": "pessoa andando de roupa vermelha",
  "results": [
    {
      "frame_id": "frame_01432",
      "timestamp": "00:23:52",
      "score": 0.847,
      "detections": ["person"],
      "thumbnail_url": "/frames/frame_01432.jpg"
    },
    {
      "frame_id": "frame_01891",
      "timestamp": "00:31:31",
      "score": 0.821,
      "detections": ["person"],
      "thumbnail_url": "/frames/frame_01891.jpg"
    }
  ],
  "latency_ms": 42
}
```

</details>

---

## ✅ Funcionalidades

| # | Funcionalidade | Descrição | Status |
|---|---|---|---|
| 1 | 🎯 **Detecção YOLOv8** | YOLOv8 Nano ONNX via OpenCV DNN, sem GPU | ✅ Implementado |
| 2 | 🧬 **Embeddings CLIP** | Vetores 512-D alinhados texto-imagem | ✅ Implementado |
| 3 | 🗄️ **ChromaDB** | Vector store principal com persistência | ✅ Implementado |
| 4 | 🔄 **Fallback KNN** | Sklearn KNN + Pickle quando ChromaDB indisponível | ✅ Implementado |
| 5 | 🌐 **API FastAPI** | Endpoint `POST /search` com texto em linguagem natural | ✅ Implementado |
| 6 | 📊 **Prometheus** | Métricas: latência, requests, frames/s | ✅ Implementado |
| 7 | 📈 **Grafana** | Dashboard de monitoramento em tempo real | ✅ Implementado |
| 8 | 🐳 **Docker Compose** | Stack completa com um único comando | ✅ Implementado |
| 9 | ⚡ **Edge-ready** | Roda em CPU, sem dependência de GPU | ✅ Implementado |
| 10 | 🔁 **Busca top-K** | Retorna os K frames mais similares com score | ✅ Implementado |

---

## 📁 Estrutura de Arquivos

```
EdgeVision-RAG/
│
├── 📄 ingest.py                  # Pipeline de ingestão de vídeo/imagens
├── 📄 api.py                     # FastAPI: endpoint de busca semântica
├── 📄 detect.py                  # YOLOv8 Nano ONNX via OpenCV DNN
├── 📄 embed.py                   # CLIP ViT-B-32 embeddings
│
├── 📂 vector_store/
│   ├── chroma_store.py           # Integração ChromaDB
│   └── knn_fallback.py           # Fallback: sklearn KNN + Pickle
│
├── 📂 monitoring/
│   ├── metrics.py                # Prometheus instrumentation
│   ├── prometheus.yml            # Configuração do Prometheus
│   └── grafana_dashboard.json    # Dashboard pré-configurado
│
├── 📂 models/
│   └── yolov8n.onnx              # Modelo YOLOv8 Nano (ONNX)
│
├── 📂 frames/                    # Frames extraídos para serving
├── 📄 docker-compose.yml         # Prometheus + Grafana
├── 📄 requirements.txt           # Dependências Python
├── 📄 .env.example               # Variáveis de configuração
└── 📄 README.md                  # Documentação
```

---

## 📊 Monitoramento — Grafana Dashboard

<details>
<summary><b>📈 Métricas disponíveis</b></summary>

| Métrica | Tipo | Descrição |
|---|---|---|
| `edgevision_frames_processed_total` | Counter | Total de frames processados |
| `edgevision_detections_total` | Counter | Total de objetos detectados |
| `edgevision_search_latency_seconds` | Histogram | Latência das buscas na API |
| `edgevision_embed_latency_seconds` | Histogram | Tempo de geração de embeddings |
| `edgevision_api_requests_total` | Counter | Total de requisições à API |
| `edgevision_vector_store_size` | Gauge | Número de vetores no ChromaDB |

```
# Acesse os dashboards em:
Grafana:    http://localhost:3000
Prometheus: http://localhost:9090
API Docs:   http://localhost:8000/docs
```

</details>

---

## 👨‍💻 Autor

<div align="center">

| | |
|---|---|
| **Nome** | Fabio Piassi |
| **Formação** | Física · Ciência de Dados · IA · DevSecOps |
| **Especialidade** | Visão Computacional · RAG · FastAPI · Cloud |
| **Localização** | Volta Redonda — RJ 🇧🇷 |
| **GitHub** | [@fassir](https://github.com/fassir) |

[![GitHub](https://img.shields.io/badge/GitHub-fassir-1F9BD4?style=for-the-badge&logo=github&logoColor=white)](https://github.com/fassir)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Fabio_Piassi-2E75B6?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/fassir)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16265F,50:2E75B6,100:1F9BD4&height=120&section=footer&fontSize=14&fontColor=ffffff&text=EdgeVision-RAG+·+by+Fabio+Piassi&fontAlignY=65" />

*"Ver é entender — e agora as máquinas entendem o que você descreve."*

</div>
