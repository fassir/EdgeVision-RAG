import cv2
import numpy as np

class YOLOv8ONNX:
    """
    Motor de Inferência Otimizado para YOLOv8 usando ONNX Runtime via OpenCV.
    
    Esta classe encapsula toda a complexidade de carregar o modelo neural,
    preparar a imagem (pré-processamento) e interpretar os resultados (pós-processamento).
    
    Atributos:
        net (cv2.dnn.Net): Objeto da rede neural carregada no OpenCV.
        img_size (tuple): Dimensão de entrada que o modelo YOLO espera (640, 640).
        classes (dict): Mapeamento de ID numérico (ex: 0) para nome da classe (ex: 'person').
    """
    
    def __init__(self, onnx_model_path, device="cuda"):
        """
        Inicializa o motor de inferência.

        Args:
            onnx_model_path (str): Caminho absoluto ou relativo para o arquivo .onnx.
            device (str): Backend de execução. 'cuda' para GPU (NVIDIA) ou 'cpu'.
                          Nota: Em muitos casos, OpenCV DNN na CPU é extremamente rápido e suficiente.
        """
        try:
            self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar modelo ONNX via OpenCV: {e}")
            
        if device == "cuda":
            print("Tentando usar backend CUDA no OpenCV DNN...")
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except:
                print("CUDA não disponível para OpenCV, usando CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.img_size = (640, 640)  # Tamanho padrão de entrada do YOLOv8

        # Nomes das classes do dataset COCO (padrão em modelos pré-treinados YOLO)
        # ID 0 = 'person' é o mais importante para este projeto.
        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    def preprocess(self, img):
        """
        Prepara a imagem crua (Raw Frame) para entrar na Rede Neural.
        
        A rede não aceita imagens de qualquer tamanho. Precisamos:
        1. Redimensionar para 640x640.
        2. Normalizar os pixels (de 0-255 para 0-1).
        3. Reorganizar os canais de cor (HWC -> CHW).
        
        Args:
            img (numpy.ndarray): Frame original capturado do vídeo (BGR).
            
        Returns:
            blob: Um objeto "blob" do OpenCV pronto para inferência (Batch of Images).
        """
        self.img_height, self.img_width = img.shape[:2]
        
        # Cria um blob 4D da imagem.
        # Definição: blob = cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)
        # scalefactor=1/255.0 reduz os pixels float para 0.0-1.0
        # swapRB=True converte BGR (Padrão OpenCV) para RGB (Padrão Neural Nets)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.img_size[1], self.img_size[0]), swapRB=True, crop=False)
        return blob

    def postprocess(self, outputs, conf_threshold=0.5, iou_threshold=0.5):
        """
        Interpreta a saída bruta (matriz de tensores) da rede neural.
        
        O YOLO retorna milhares de caixas possíveis (ex: 8400 caixas). A maioria está errada
        ou duplicada. Esta função filtra o 'lixo' usando Non-Max Suppression (NMS).
        
        Args:
            outputs (list): Saída direta do `net.forward()`.
            conf_threshold (float): Confiança mínima para aceitar uma detecção (0.0 a 1.0).
            iou_threshold (float): Limiar de sobreposição para considerar duas caixas como a mesma (Intersection over Union).
            
        Retorna:
            tuple: (boxes, scores, class_ids)
                   boxes: Coordenadas finais [x1, y1, x2, y2] das caixas.
                   scores: Níveis de certeza.
                   class_ids: Qual objeto é (0=Pessoa).
        """
        # A saída do YOLOv8 ONNX tem shape (1, 84, 8400)
        # 84 canais = 4 coordenadas (center_x, center_y, w, h) + 80 classes
        # 8400 = total de âncoras/previsões
        predictions = np.squeeze(outputs[0]).T  # Transpõe para (8400, 84)

        # Filtra previsões com baixa confiança
        scores = np.max(predictions[:, 4:], axis=1) # Pega a maior probabilidade entre as 80 classes
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        
        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Extrai coordenadas das caixas
        # YOLO devolve: x_centro, y_centro, largura, altura (normalizados ou em pixels dependendo da versão)
        boxes = predictions[:, :4] 
        
        # Normaliza em relação ao tamanho de entrada da rede (640)
        input_shape = np.array([self.img_size[1], self.img_size[0], self.img_size[1], self.img_size[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32) 
        # Escala para o tamanho original da imagem
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Converte formato (centro, w, h) -> (canto_sup_esq, canto_inf_dir)
        xyxy = np.copy(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 (esquerdo)
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 (topo)
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 (direito)
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 (base)

        # Aplica NMS (Non-Max Suppression)
        # Remove caixas sobrepostas que se referem ao mesmo objeto
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        
        return xyxy[indices], scores[indices], class_ids[indices]

    def draw_detections(self, img, boxes, scores, class_ids):
        """
        Desenha as caixas detectadas na imagem original.
        
        Args:
            img (numpy.ndarray): Frame original.
            boxes, scores, class_ids: Resultados do postprocess().
            
        Retorna:
            tuple: (imagem_anotada, lista_de_deteccoes_limpa)
        """
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Filtro de Negócio: Para este projeto de RAG de 'pessoas', ignoramos carros, gatos, etc.
            # ID 0 = Person no COCO Dataset
            if class_id != 0:
                continue
                
            x1, y1, x2, y2 = box.astype(int)
            detections.append((x1, y1, x2, y2, score))

            label = f"{self.classes[class_id]}: {score:.2f}"
            
            # Desenha retângulo verde
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Escreve o label
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img, detections

    def infer(self, img):
        """
        Executa o fluxo completo de inferência em uma imagem.
        
        Args:
            img (numpy.ndarray): Frame de vídeo.
            
        Retorna:
            tuple: (imagem_processada, lista_deteccoes)
        """
        input_tensor = self.preprocess(img)
        self.net.setInput(input_tensor)
        outputs = self.net.forward()
        
        # Garante que a saída seja uma lista (compatibilidade entre versões OpenCV)
        if not isinstance(outputs, list):
            outputs = [outputs]

        boxes, scores, class_ids = self.postprocess(outputs)
        return self.draw_detections(img, boxes, scores, class_ids)
            
        boxes, scores, class_ids = self.postprocess(outputs)
        return self.draw_detections(img, boxes, scores, class_ids)
