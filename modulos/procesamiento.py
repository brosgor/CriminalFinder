import os
import pickle
import logging
from typing import Dict, List, Any
import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_embeddings(dataset_dir: str, output_path: str, model: str = "hog") -> None:
    """
    Construye la base de embeddings a partir de imágenes en dataset_dir usando OpenCV.

    Args:
        dataset_dir: Directorio con imágenes de criminales
        output_path: Ruta donde guardar el archivo pickle
        model: Modelo de detección ('hog' o 'cnn') - ignorado, usa OpenCV Haar Cascade
    """
    names: List[str] = []
    encodings: List[List[float]] = []
    images: List[str] = []
    meta: Dict[str, Any] = {"model": "opencv_haar", "total_images": 0, "total_faces": 0}

    # Cargar detector de rostros de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Crear reconocedor LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Recorrer el directorio
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                try:
                    # Cargar imagen
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"No se pudo cargar la imagen: {image_path}")
                        continue

                    # Convertir a escala de grises
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detectar rostros
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) == 0:
                        logger.warning(f"No se detectaron rostros en: {image_path}")
                        continue
                    elif len(faces) > 1:
                        logger.warning(f"Múltiples rostros detectados en: {image_path}, usando el primero")

                    # Tomar el primer rostro
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Redimensionar a tamaño fijo
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Usar el histograma como "embedding"
                    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                    hist = hist.flatten().tolist()

                    # Determinar nombre: del archivo o carpeta padre
                    if os.path.basename(root) != os.path.basename(dataset_dir):
                        name = os.path.basename(root)
                    else:
                        name = os.path.splitext(file)[0]

                    names.append(name)
                    encodings.append(hist)
                    images.append(image_path)
                    meta["total_faces"] += 1

                except Exception as e:
                    logger.error(f"Error procesando {image_path}: {e}")
                    continue

                meta["total_images"] += 1

    # Guardar con pickle
    data = {
        "names": names,
        "encodings": encodings,
        "images": images,
        "meta": meta
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    logger.info(f"Embeddings guardados en {output_path}. Total rostros: {meta['total_faces']}")

# Barra de progreso se maneja implícitamente con tqdm si se usa en un loop, pero aquí no es necesario ya que os.walk no tiene progreso directo
