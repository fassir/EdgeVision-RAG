import uvicorn
import os
import sys

# Adiciona o diretório atual ao path do Python para encontrar o módulo 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

"""
Script de Inicialização da API.

Este arquivo é o ponto de entrada para rodar o servidor web (FastAPI).
Ele usa o Uvicorn como servidor de aplicação ASGI.

Comando:
    python run_api.py
"""

if __name__ == "__main__":
    print("Iniciando Servidor API em http://localhost:8000")
    print("Documentação interativa disponível em http://localhost:8000/docs")
    
    # Roda o Uvicorn
    # host="0.0.0.0" permite acesso externo (outros computadores na rede)
    # reload=False para produção (True apenas para dev)
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
