# Diagnóstico de Lesiones Cutáneas con Deep Learning

Clasificación multimodal de 7 tipos de lesiones cutáneas usando el dataset **HAM10000**,
combinando imágenes dermatoscópicas y metadatos clínicos mediante técnicas de fusión de modalidades.

**Autor:** Rubén Cerezo

---

## Resultados

| Modelo | Datos de entrada | Accuracy | Mejora vs baseline |
|---|---|---|---|
| Baseline ZeroR | Clase mayoritaria siempre | 66.93% | — |
| Red Densa | Solo metadatos clínicos | 70.39% | +3.46% |
| CNN (Transfer Learning) | Solo imágenes | 72.39% | +5.46% |
| Late Fusion (SVM) | Imágenes + metadatos | 78.64% | +11.71% |
| **Early Fusion** | **Imágenes + metadatos** | **78.44%** | **+11.51%** |

> El baseline ZeroR predice siempre la clase mayoritaria (nevus, ~67%). Es el punto de referencia mínimo:
> cualquier modelo que no lo supere no ha aprendido nada útil.

Los modelos de fusión (Early y Late) superan en ~11 puntos al baseline, confirmando que combinar
imágenes con metadatos clínicos es la estrategia más efectiva para este problema.

---

## Dataset

**HAM10000** — 10.015 imágenes dermatoscópicas de lesiones cutáneas etiquetadas por dermatólogos,
distribuidas en 7 clases diagnósticas:

| Código | Diagnóstico | Casos |
|---|---|---|
| `nv` | Nevus melanocítico (lunar benigno) | 6.705 |
| `mel` | Melanoma | 1.113 |
| `bkl` | Lesión queratinocítica benigna | 1.099 |
| `bcc` | Carcinoma basocelular | 514 |
| `akiec` | Queratosis actínica | 327 |
| `vasc` | Lesión vascular | 142 |
| `df` | Dermatofibroma | 115 |

> El dataset presenta un fuerte desbalanceo de clases (nevus ~67%). Se aplica `stratify` en los
> splits y `class_weight='balanced'` durante el entrenamiento para compensarlo.

Para descargar los datos, consulta [`data/README.md`](data/README.md).

---

## Estructura del proyecto

```
KC-Deeplearning-/
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    ← EDA: distribución de clases, estadísticas, visualizaciones
│   ├── 02_data_preparation.ipynb        ← Pipeline de preprocesado reutilizable
│   ├── 03_model_tabular.ipynb           ← Modelo baseline con solo metadatos
│   ├── 04_model_cnn.ipynb               ← CNN con Transfer Learning (MobileNetV3Large)
│   ├── 05_model_early_fusion.ipynb      ← Early Fusion (mejor modelo)
│   ├── 06_model_late_fusion.ipynb       ← Late Fusion con SVM
│   └── 07_metrics_and_conclusions.ipynb ← Comparativa final y conclusiones
│
├── data/
│   ├── raw/                             ← CSVs del dataset (no incluidos, ver data/README.md)
│   └── README.md                        ← Instrucciones de descarga
│
├── models/                              ← Modelos entrenados (.h5 / .keras, no incluidos en git)
├── utils.py                             ← Pipeline de preprocesado centralizado
├── docs/                                ← Enunciado, guion y descripción del dataset
└── requirements.txt                     ← Dependencias con versiones exactas
```

---

## Cómo ejecutar

### 1. Clonar el repositorio

```bash
git clone https://github.com/Rcerezo-dev/KC-Deeplearning-.git
cd KC-Deeplearning-
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar los datos

Sigue las instrucciones de [`data/README.md`](data/README.md) para obtener los CSVs
y colócalos en `data/raw/`.

### 4. Ejecutar los notebooks en orden

Los notebooks están numerados. Cada uno carga sus propios datos a través de `utils.py`
y guarda el modelo resultante en `models/`. Se recomienda ejecutarlos en orden:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

El notebook `07` requiere que los notebooks `03`–`06` se hayan ejecutado al menos una vez
para cargar los modelos guardados.

---

## Arquitecturas

### Early Fusion (mejor modelo)

Dos ramas procesan imagen y metadatos en paralelo y se concatenan antes de la clasificación.
Al unirse durante el entrenamiento, la red aprende interacciones entre ambas modalidades.

```
Input imágenes (28×28×3)              Input tabular (17 features)
        │                                        │
  Resizing(224×224)                        Dense(32)
  MobileNetV3Large (frozen)               Dropout(0.3)
  GlobalAveragePooling2D                  Dense(32)
        │                                     │
        └──────────── Concatenate ────────────┘
                           │
                      Dense(256) + Dropout(0.5)
                      Dense(128)
                      Dense(7, softmax)
```

### CNN con Transfer Learning

Backbone **MobileNetV3Large** preentrenado en ImageNet. Entrenamiento en dos fases:
1. Solo la cabeza densa (backbone congelado, LR=1e-3)
2. Fine-tuning del último 30% del backbone (LR=1e-5)

```
Input (28×28×3)
→ Resizing(224×224)
→ MobileNetV3Large (ImageNet)
→ GlobalAveragePooling2D
→ Dense(512) + Dropout(0.4)
→ Dense(128)
→ Dense(7, softmax)
```

### Late Fusion (SVM)

Dos modelos independientes (CNN + Red Densa tabular) cuyas probabilidades de salida
se concatenan y se pasan a un **SVM con kernel RBF** que aprende la combinación óptima.

```
CNN → prob(7) ──┐
                ├─ Concatenate(14) → SVM(rbf) → pred(7)
Tabular → prob(7) ─┘
```

---

## Decisiones técnicas clave

- **Split 70/15/15** con `stratify` — tres conjuntos fijos para evitar data leakage entre
  la validación usada en EarlyStopping y la evaluación final.
- **Preprocesado centralizado en `utils.py`** — todos los notebooks comparten el mismo pipeline:
  alineamiento por `image_id`, imputación de nulos, normalización per-canal con StandardScaler.
- **Normalización per-canal** — `StandardScaler` ajustado solo sobre train para cada canal RGB,
  evitando que información del test influya en la normalización.
- **`class_weight='balanced'`** — compensa el fuerte desbalanceo (nevus ~67%) penalizando más
  los errores en clases minoritarias como melanoma.

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-yellowgreen)

- **Deep Learning:** TensorFlow · Keras
- **Transfer Learning:** MobileNetV3Large (ImageNet)
- **Fusión multimodal:** Early Fusion (Functional API) · Late Fusion (SVM)
- **Datos:** pandas · NumPy · scikit-learn
- **Visualización:** matplotlib

---

## Conclusiones principales

1. **Los metadatos solos son insuficientes** — El modelo tabular alcanza un 70%, apenas por encima
   del baseline del 67%. El contexto clínico aporta señal, pero no es suficiente sin imagen.

2. **Transfer Learning supera una CNN desde cero** — MobileNetV3Large, preentrenado en ImageNet,
   extrae features visuales (bordes, texturas) que una CNN entrenada con imágenes 28×28
   desde cero no puede aprender por falta de resolución y datos.

3. **La fusión de modalidades es la estrategia ganadora** — Tanto Early como Late Fusion superan
   en ~6 puntos a la CNN sola, confirmando que imagen y metadatos se complementan.

4. **El desbalanceo de clases sigue siendo el principal reto** — Aunque `class_weight` mejora el
   recall en clases minoritarias, el melanoma sigue siendo difícil de detectar correctamente,
   lo cual tiene implicaciones clínicas relevantes (un falso negativo puede ser grave).
