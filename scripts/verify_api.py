import requests
import sys
import time

"""
Script de Validação de API (Validation/Quality Assurance).

Diferente do `test_api.py`, este script foca em dar feedback visual detalhado com emojis
sobre a saúde do sistema e a qualidade (Sanity Check) dos resultados da busca.
"""

def check_api():
    base_url = "http://localhost:8000"
    
    print(f"🔍 1. Verificando conectividade em {base_url}...")
    try:
        # Check Stats (Monitoramento)
        r = requests.get(f"{base_url}/stats")
        if r.status_code != 200:
            print(f"❌ Falha em /stats: {r.status_code}")
            return False
        
        stats = r.json()
        count = stats.get("total_vectors", 0)
        print(f"✅ API Online! Total de vetores no banco: {count}")
        
        if count == 0:
            print("⚠️ Aviso: Banco de dados está vazio na memória da API.")
        
        # Check Search (Funcional)
        query = "person"
        print(f"🔍 2. Testando busca semântica para: '{query}'...")
        r = requests.get(f"{base_url}/search", params={"q": query, "limit": 3})
        
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            print(f"✅ Busca retornou {len(results)} resultados.")
            for i, res in enumerate(results):
                timestamp = res.get('timestamp', 'N/A')
                dist = res.get('distance', 0.0)
                # Similaridade = 1 - Distância (aproximado para Cosine)
                print(f"   Result {i+1}: Tempo {timestamp:.2f}s (Distância: {dist:.4f})")
            return True
        else:
            print(f"❌ Falha na busca: {r.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Não foi possível conectar na API. Ela está rodando? (python run_api.py)")
        return False

if __name__ == "__main__":
    success = check_api()
    if not success:
        sys.exit(1)
