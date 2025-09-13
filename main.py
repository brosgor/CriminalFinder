import argparse
import logging
import os
import sys
from modulos.procesamiento import build_embeddings
from modulos.deteccion import RealtimeRecognizer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Sistema de reconocimiento facial modular")
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operación')

    # Subparser para modo offline
    offline_parser = subparsers.add_parser('offline', help='Construir base de embeddings')
    offline_parser.add_argument('--input', required=True, help='Directorio de imágenes de criminales')
    offline_parser.add_argument('--output', required=True, help='Archivo de salida para embeddings')
    offline_parser.add_argument('--model', choices=['hog', 'cnn'], default='hog', help='Modelo de detección')

    # Subparser para modo realtime
    realtime_parser = subparsers.add_parser('realtime', help='Reconocimiento en tiempo real')
    realtime_parser.add_argument('--db', required=True, help='Archivo de base de embeddings')
    realtime_parser.add_argument('--threshold', type=float, default=0.6, help='Umbral de similitud')
    realtime_parser.add_argument('--detector', choices=['hog', 'cnn'], default='hog', help='Modelo de detección')
    realtime_parser.add_argument('--camera', type=int, default=0, help='Índice de la cámara')

    args = parser.parse_args()

    if args.mode == 'offline':
        if not os.path.isdir(args.input):
            logging.error(f"El directorio {args.input} no existe")
            sys.exit(1)
        try:
            build_embeddings(args.input, args.output, args.model)
            logging.info("Base de embeddings construida exitosamente")
        except Exception as e:
            logging.error(f"Error al construir embeddings: {e}")
            sys.exit(1)
    elif args.mode == 'realtime':
        if not os.path.isfile(args.db):
            logging.error(f"El archivo {args.db} no existe")
            sys.exit(1)
        try:
            recognizer = RealtimeRecognizer(args.db, args.threshold, args.detector)
            recognizer.recognize_from_webcam(args.camera)
        except Exception as e:
            logging.error(f"Error en reconocimiento en tiempo real: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
