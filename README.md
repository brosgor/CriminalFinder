# Proyecto de Reconocimiento Facial

## Descripción
Este proyecto implementa un sistema de reconocimiento facial modular para el curso de Computación Visual (2025-II, profesora Aura María Forero Pachón). Incluye dos módulos principales: uno offline para construir una base de datos de embeddings a partir de imágenes de "criminales", y otro en tiempo real para reconocer rostros usando la webcam.

## Requisitos de Sistema
- Python 3.10+
- Webcam funcional
- Sistema operativo: Linux, macOS o Windows

## Instalación

### Crear y activar entorno virtual
```bash
# Crear venv
python -m venv .venv

# Activar venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Notas de Instalación para face_recognition/dlib
La biblioteca `face_recognition` depende de `dlib`, que requiere compilación. En algunos sistemas, puede necesitar dependencias adicionales:

- **Ubuntu/Debian**: `sudo apt-get install build-essential cmake libgtk-3-dev libboost-all-dev`
- **macOS**: Instalar Xcode Command Line Tools
- **Windows**: Instalar Visual Studio Build Tools

Si `face_recognition` no se instala correctamente, el proyecto incluye un fallback básico usando OpenCV, pero con menor precisión.

Documentación oficial: [face_recognition](https://github.com/ageitgey/face_recognition) y [dlib](http://dlib.net/)

## Uso

### 1. Preparar imágenes
Coloca imágenes de personas en el directorio `criminales/`. Puedes usar:
- Archivos individuales: `criminales/Juan_Perez.jpg`
- Carpetas por persona: `criminales/Juan_Perez/1.jpg`, `criminales/Juan_Perez/2.jpg`

### 2. Construir base de embeddings
```bash
python main.py offline --input criminales --output embeddings_criminales.pkl --model hog
```

### 3. Ejecutar reconocimiento en tiempo real
```bash
python main.py realtime --db embeddings_criminales.pkl --threshold 0.6 --detector hog --camera 0
```

## Parámetros Recomendados
- `scale`: 0.25 para buen rendimiento en CPU
- `detector`: "hog" para CPU, "cnn" si tienes GPU compatible con CUDA
- `threshold`: 0.6 por defecto, ajustar según necesidades (menor valor = más estricto)

## Métricas Esperadas
- ≥15 FPS en CPU moderna con scale=0.25
- Precisión depende de la calidad de las imágenes de referencia

## Ética y Privacidad
Este proyecto es únicamente para fines académicos. Los datos se procesan localmente sin enviar información a servicios externos.

## Troubleshooting
- **No detecta cámara**: Verificar que la webcam esté conectada y no en uso por otra aplicación
- **Error instalando dlib**: Seguir las instrucciones de compilación específicas del sistema
- **Cero rostros detectados**: Verificar iluminación y calidad de imágenes
- **Tuning de threshold**: Probar valores entre 0.4-0.8 según el caso de uso

## Estructura del Proyecto
```
proyecto_reconocimiento/
├── main.py                 # Punto de entrada
├── modulos/
│   ├── procesamiento.py    # Módulo offline
│   └── deteccion.py        # Módulo tiempo real
├── criminales/             # Imágenes de referencia
├── embeddings_criminales.pkl # Base de datos generada
├── requirements.txt        # Dependencias
└── README.md               # Este archivo
```
