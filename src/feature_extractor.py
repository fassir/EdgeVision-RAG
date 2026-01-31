from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

class FeatureExtractor:
    """
    Wrapper para o modelo CLIP (Contrastive Language-Image Pre-Training).
    
    Esta classe é o coração da "inteligência multimodal" do sistema.
    Ela permite converter tanto imagens quanto textos para o mesmo espaço vetorial (embeddings),
    o que possibilita comparar a similaridade entre uma descrição textual ("homem correndo")
    e uma imagem (pixels de um frame).
    
    Atributos:
        model (SentenceTransformer): Instância do modelo carregado (ex: ViT-B-32).
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        """
        Carrega o modelo CLIP na memória.
        
        Args:
            model_name (str): Nome do modelo no HuggingFace Hub.
                              'clip-ViT-B-32' é um bom balanço entre performance (velocidade) e precisão.
        """
        # "clip-ViT-B-32" mapeia imagens e texto para um vetor compartilhado de 512 dimensões.
        print(f"Loading Feature Extractor model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Feature Extractor loaded.")

    def extract_image_embedding(self, image_input):
        """
        Gera um vetor (embedding) a partir de uma imagem.

        Args:
            image_input (numpy.ndarray | PIL.Image.Image): 
                - Se for numpy.ndarray (OpenCV), assume BGR e converte para RGB.
                - Se for PIL.Image, usa diretamente.

        Retorna:
            numpy.ndarray: Vetor de 512 números (float32).
        """
        # Se for um array numpy (vdo CV2), converte BGR -> RGB
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input[:, :, ::-1])
        else:
            # Assume que já é um objeto PIL Image (ex: vindo de upload da API)
            image = image_input
        
        # O método .encode() do SentenceTransformers trata todo o pipeline:
        # Resize -> Normalize -> Forward Pass -> Embedding
        embedding = self.model.encode(image)
        return embedding

    def extract_text_embedding(self, text: str):
        """
        Gera um vetor (embedding) a partir de uma query de texto.
        
        Usado na hora da busca para converter a pergunta do usuário ("homem de boné")
        em um vetor numérico compatível com os vetores das imagens.

        Args:
            text (str): Texto da busca.

        Retorna:
            numpy.ndarray: Vetor de 512 números (float32).
        """
        return self.model.encode(text)
