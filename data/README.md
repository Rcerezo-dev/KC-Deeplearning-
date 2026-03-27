# Datos del proyecto

Los archivos CSV del dataset HAM10000 no están incluidos en el repositorio por su tamaño.

## Descarga

### Opción 1 — Kaggle
1. Accede a [HAM10000 en Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
2. Descarga `HAM10000_metadata.csv`
3. El archivo de imágenes aplanadas (`hnmist_28_28_RGB.csv`) fue generado a partir de las imágenes originales en formato 28×28 RGB

### Opción 2 — ISIC Archive
- https://www.isic-archive.com/

## Archivos esperados

Coloca los archivos en `data/raw/`:

```
data/
└── raw/
    ├── HAM10000_metadata.csv   (~5 MB)
    └── hnmist_28_28_RGB.csv    (~500 MB)
```
