import pickle
import logging
from typing import List, Dict, Any
import cv2
import numpy as np
import time

logger = logging.getLogger(__name__)

class RealtimeRecognizer:
    def __init__(self, db_path: str, threshold: float = 0.6, detector: str = "hog"):
        """
        Inicializa el reconocedor en tiempo real usando OpenCV.

        Args:
            db_path: Ruta al archivo de base de embeddings
            threshold: Umbral de similitud para matches
            detector: Modelo de detección (ignorado, usa OpenCV)
        """
        self.threshold = threshold
        self.detector = detector

        # Cargar base de datos
        with open(db_path, 'rb') as f:
            self.data: Dict[str, Any] = pickle.load(f)

        self.known_names: List[str] = self.data["names"]
        self.known_encodings: List[np.ndarray] = [np.array(enc) for enc in self.data["encodings"]]

        # Cargar detector de rostros de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        logger.info(f"Base cargada: {len(self.known_names)} rostros conocidos")

    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compara dos histogramas y retorna la distancia (0 = idénticos, 1 = completamente diferentes)
        """
        # Usar correlación de histogramas
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Convertir correlación a distancia (invertir y normalizar)
        distance = 1.0 - correlation
        return distance

    def recognize_from_webcam(self, camera_index: int = 0, display_fps: bool = True, scale: float = 0.25):
        """
        Ejecuta el reconocimiento en tiempo real desde la webcam.

        Args:
            camera_index: Índice de la cámara
            display_fps: Mostrar FPS
            scale: Factor de escala para redimensionar frames
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("No se pudo abrir la cámara")
            return

        logger.info("Presiona 'q' para salir")

        prev_time = time.time()
        fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("No se pudo leer el frame")
                    break

                # Convertir a escala de grises para detección
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detectar rostros
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                # Reconocer cada rostro
                names = []
                for (x, y, w, h) in faces:
                    # Extraer ROI del rostro
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Calcular histograma
                    current_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                    current_hist = current_hist.flatten()

                    # Comparar con rostros conocidos
                    name = "Desconocido"
                    min_distance = float('inf')
                    
                    for i, known_encoding in enumerate(self.known_encodings):
                        known_hist = np.array(known_encoding, dtype=np.float32)
                        distance = self._compare_histograms(current_hist, known_hist)
                        
                        if distance < min_distance and distance < self.threshold:
                            min_distance = distance
                            name = self.known_names[i]

                    names.append(name)

                # Dibujar resultados
                for (x, y, w, h), name in zip(faces, names):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Calcular y mostrar FPS
                if display_fps:
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
                    prev_time = current_time
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow('Reconocimiento Facial', frame)

                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupción detectada")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Sesión terminada")
