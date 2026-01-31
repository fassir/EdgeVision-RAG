import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os
import time

"""
Dashboard de Investigação EdgeVision RAG.

Este script é uma interface visual (Frontend) construída com Streamlit.
Ele atua como o painel de controle para o operador de segurança ou investigador.

Funcionalidades:
1. Conecta-se à API REST local.
2. Permite buscas textuais ("homem correndo").
3. Permite buscas visuais (Upload de imagem -> Encontrar similar).
4. Exibe os resultados com timestamp e player de vídeo posicionado.

Como rodar:
    streamlit run dashboard.py
"""

# Configuração da Página
st.set_page_config(
    page_title="EdgeVision RAG Dashboard",
    page_icon="👁️",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("👁️ EdgeVision RAG - Dashboard de Investigação")
st.markdown("Busca semântica inteligente em vídeos de segurança.")

# Sidebar com Status do Sistema
with st.sidebar:
    st.header("Status do Sistema")
    if st.button("Verificar Conexão API"):
        try:
            r = requests.get(f"{API_URL}/stats")
            if r.status_code == 200:
                stats = r.json()
                st.success("API Online! 🟢")
                st.metric("Total de Vetores", stats.get("total_vectors", 0))
                st.metric("Backend", stats.get("backend_type", "Unknown"))
            else:
                st.error("Erro na API 🔴")
        except:
            st.error("API Offline 🔴")
    
    st.info("Vídeos devem estar na pasta `data/` para serem reproduzidos.")

# Abas Principais
tab1, tab2 = st.tabs(["🔍 Busca por Texto", "🖼️ Busca por Imagem"])

# --- TAB 1: BUSCA POR TEXTO ---
with tab1:
    st.subheader("Descreva o que você procura")
    query_text = st.text_input("Ex: 'pessoa de mochila vermelha', 'carro azul', 'crianca correndo'", "")
    
    if st.button("Buscar por Texto"):
        if query_text:
            with st.spinner("Pesquisando no cérebro digital..."):
                try:
                    r = requests.get(f"{API_URL}/search", params={"q": query_text, "limit": 6})
                    if r.status_code == 200:
                        results = r.json().get("results", [])
                        st.subheader(f"Encontrados {len(results)} resultados")
                        
                        # Grid de Resultados
                        cols = st.columns(3)
                        for idx, res in enumerate(results):
                            with cols[idx % 3]:
                                timestamp = res['timestamp']
                                minute = int(timestamp // 60)
                                second = int(timestamp % 60)
                                time_str = f"{minute:02d}:{second:02d}"
                                
                                st.markdown(f"**⏱️ {time_str}**")
                                st.caption(f"Score Distância: {res['distance']:.4f}")
                                
                                # Tenta mostrar o vídeo no momento exato
                                source_video = res.get('metadata_completo', {}).get('source', '')
                                
                                # Se o source for apenas um numero (webcam), ignora
                                if source_video and not source_video.isdigit():
                                    # Corrige caminho relativo se necessario
                                    # run_pipeline salva paths relativos ou absolutos dependendo de como rodou
                                    # Vamos tentar achar o arquivo em 'data/' se ele for apenas um nome
                                    video_name = os.path.basename(source_video)
                                    local_path = os.path.join("data", video_name)
                                    
                                    if os.path.exists(local_path):
                                        st.video(local_path, start_time=int(timestamp))
                                    elif os.path.exists(source_video):
                                         st.video(source_video, start_time=int(timestamp))
                                    else:
                                        st.warning(f"Vídeo original não encontrado: {video_name}")
                                else:
                                    st.info("Origem: Webcam/Live Stream")
                                    
                                st.divider()
                    else:
                        st.error("Erro na busca.")
                except Exception as e:
                    st.error(f"Erro de conexão: {e}")

# --- TAB 2: BUSCA POR IMAGEM ---
with tab2:
    st.subheader("Upload de Imagem Referência")
    uploaded_file = st.file_uploader("Envie uma foto da pessoa/objeto que procura", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Imagem Query", width=200)
        
        with col2:
            if st.button("Buscar por Similaridade Visual"):
                with st.spinner("Analisando pixels..."):
                    try:
                        # Re-ler o arquivo para enviar
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        r = requests.post(f"{API_URL}/search_image", files=files, params={"limit": 6})
                        
                        if r.status_code == 200:
                            results = r.json().get("results", [])
                            st.success(f"Encontrados {len(results)} matches visuais!")
                        else:
                            st.error(f"Erro na API: {r.text}")
                            results = []
                    except Exception as e:
                        st.error(f"Erro: {e}")
                        results = []

        if 'results' in locals() and results:
            st.markdown("---")
            st.subheader("Resultados Visuais")
            res_cols = st.columns(3)
            for idx, res in enumerate(results):
                with res_cols[idx % 3]:
                    timestamp = res['timestamp']
                    min_sec = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                    st.markdown(f"#### ⏱️ {min_sec}")
                    st.progress(max(0.0, 1.0 - res['distance']), text=f"Similaridade {(1-res['distance'])*100:.1f}%")
                    
                    # Video Player Logic (Duplicada para esta aba)
                    source = res.get('source_video', '')
                    if source and not source.isdigit():
                        v_name = os.path.basename(source)
                        possible_paths = [
                            os.path.join("data", v_name),
                            source
                        ]
                        found = False
                        for p in possible_paths:
                            if os.path.exists(p):
                                st.video(p, start_time=int(timestamp))
                                found = True
                                break
                        if not found:
                             st.warning(f"Arquivo {v_name} não acessível.")
                    st.divider()

# Footer
st.markdown("---")
st.caption("EdgeVision RAG v1.0 | Dashboard Built with Streamlit")