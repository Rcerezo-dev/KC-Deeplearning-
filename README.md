# Diagnóstico de Lesiones Cutáneas con Deep Learning

Clasificación multimodal de 7 tipos de lesiones cutáneas usando el dataset **HAM10000**,
combinando imágenes dermatoscópicas y metadatos clínicos mediante técnicas de fusión de modalidades.

**Autor:** Rubén Cerezo

---

## Resultados

| Modelo | Datos de entrada | Accuracy |
|---|---|---|
| Red Densa (baseline) | Solo metadatos clínicos | 69% |
| CNN | Solo imágenes | 74% |
| Late Fusion | Imágenes + metadatos | 75% |
| **Early Fusion** | **Imágenes + metadatos** | **~80%** |

El modelo de Early Fusion es el mejor del proyecto: al combinar ambas modalidades en una
arquitectura unificada, aprende las interacciones entre los datos visuales y el contexto
clínico del paciente (edad, sexo, localización de la lesión).

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

> El dataset presenta un fuerte desbalanceo de clases (nevus representa el ~67% de los datos),
> lo que hace esencial el uso de `stratify` en los splits de datos.

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
│   ├── 04_model_cnn.ipynb               ← CNN con solo imágenes
│   ├── 05_model_early_fusion.ipynb      ← Early Fusion (mejor modelo)
│   ├── 06_model_late_fusion.ipynb       ← Late Fusion
│   └── 07_metrics_and_conclusions.ipynb ← Comparativa final y conclusiones
│
├── data/
│   ├── raw/                             ← CSVs del dataset (no incluidos, ver data/README.md)
│   └── README.md                        ← Instrucciones de descarga
│
├── models/                              ← Modelos entrenados (.h5 / .keras, no incluidos en git)
├── docs/                                ← Enunciado, guion y descripción del dataset
├── experiments/                         ← Modelos experimentales descartados
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

Los notebooks están numerados y son independientes entre sí (cada uno carga sus propios datos),
pero se recomienda ejecutarlos en orden:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

---

## Arquitecturas

### Early Fusion (mejor modelo)
```
Input imágenes (28×28×3)         Input tabular (edad, sexo, localización)
        │                                        │
   Conv2D(32) + MaxPool                    Dense(32)
   Conv2D(64) + MaxPool                    Dense(32)
   Conv2D(128) + MaxPool                      │
        │                                     │
      Flatten ──────────── Concatenate ───────┘
                                │
                           Dropout(0.5)
                                │
                         Dense(7, softmax)
```

### CNN
```
Input (28×28×3)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten → Dense(256) + Dropout(0.5)
→ Dense(7, softmax)
```

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Keras](https://img.shields.io/badge/Keras-3.11-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-yellowgreen)

- **Deep Learning:** TensorFlow · Keras
- **Datos:** pandas · NumPy · scikit-learn
- **Imágenes:** OpenCV
- **Visualización:** matplotlib · seaborn

---

## Conclusiones principales

1. **Los metadatos solos son insuficientes** — El modelo tabular alcanza un 69%, apenas por encima
   del baseline de predecir siempre la clase mayoritaria (~67%). El contexto clínico ayuda,
   pero no es suficiente sin información visual.

2. **Las imágenes mejoran notablemente el diagnóstico** — La CNN sube al 74%, confirmando
   que la morfología visual de la lesión es la fuente de información más discriminativa.

3. **La fusión temprana supera a la tardía** — Early Fusion (~80%) supera a Late Fusion (~75%)
   porque aprende las interacciones entre imagen y metadatos durante el entrenamiento,
   en lugar de combinar predicciones independientes a posteriori.

4. **El desbalanceo de clases es el principal reto pendiente** — Los modelos predicen bien
   la clase mayoritaria (nevus) pero tienen dificultades con clases minoritarias como
   el melanoma, que es precisamente el diagnóstico de mayor relevancia clínica.
