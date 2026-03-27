"""
utils.py — Pipeline de preprocesado centralizado
=================================================
Todas las funciones de carga, alineamiento, split y transformación de datos.
Importar desde cualquier notebook con:

    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    from utils import get_all_splits, get_tabular_splits
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.utils import to_categorical

# Rutas absolutas relativas a este archivo (funcionan desde cualquier directorio)
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_BASE, "data", "raw")


def load_metadata():
    """Carga el CSV de metadatos clínicos."""
    path = os.path.join(_DATA, "HAM10000_metadata.csv")
    return pd.read_csv(path)


def load_and_align():
    """
    Carga los dos CSVs y los fusiona por image_id.

    Por qué fusionar por clave y no por posición:
    Alinear por [:len(df)] asume que ambos archivos están en el mismo orden.
    Si no lo están, cada imagen se empareja con el diagnóstico incorrecto.
    La fusión por image_id garantiza el alineamiento independientemente del orden.

    Returns
    -------
    df : DataFrame con columnas de píxeles + image_id, dx, age, sex, localization
    """
    df_images = pd.read_csv(os.path.join(_DATA, "hnmist_28_28_RGB.csv"))
    metadata  = pd.read_csv(os.path.join(_DATA, "HAM10000_metadata.csv"))

    n = min(len(df_images), len(metadata))
    df_images = df_images.iloc[:n].reset_index(drop=True)
    metadata  = metadata.iloc[:n].reset_index(drop=True)

    df_images["image_id"] = metadata["image_id"]
    df = df_images.merge(
        metadata[["image_id", "dx", "age", "sex", "localization"]],
        on="image_id"
    )
    return df


def _split_70_15_15(arrays, y_encoded, random_state=42):
    """
    Divide una lista de arrays en train (70%) / val (15%) / test (15%).

    Por qué tres conjuntos en lugar de dos:
    - Train:      el modelo aprende de aquí
    - Validación: se usa durante el entrenamiento para decidir cuándo parar
                  (EarlyStopping) sin contaminar la evaluación final
    - Test:       se usa UNA SOLA VEZ al final para medir el rendimiento real

    Con solo train/test, la validación sale del train de forma aleatoria en cada
    ejecución, lo que hace los resultados no reproducibles y mezcla su rol con el test.

    Por qué stratify:
    HAM10000 tiene un fuerte desbalanceo (nevus ~67%). Sin stratify, el split
    aleatorio podría dejar muy pocos ejemplos de melanoma en el test, haciendo
    la evaluación poco fiable.
    """
    # Paso 1: separar 70% train / 30% resto
    splits_train = train_test_split(
        *arrays, y_encoded,
        test_size=0.30, random_state=random_state, stratify=y_encoded
    )
    # train_test_split devuelve los arrays INTERCALADOS:
    # [arr1_train, arr1_test, arr2_train, arr2_test, ..., y_train, y_test]
    # Por eso usamos slicing par/impar en lugar de bloques contiguos.
    n = len(arrays)
    trains   = splits_train[0::2][:n]   # índices 0, 2, 4, ... → partes train
    temps    = splits_train[1::2][:n]   # índices 1, 3, 5, ... → partes temp (30%)
    enc_temp = splits_train[-1]         # y_encoded del 30% (para stratify en paso 2)

    # Paso 2: dividir el 30% en val (15%) y test (15%)
    splits_val = train_test_split(
        *temps,
        test_size=0.50, random_state=random_state, stratify=enc_temp
    )
    vals  = splits_val[0::2][:n]
    tests = splits_val[1::2][:n]

    return trains, vals, tests


def get_tabular_splits(random_state=42):
    """
    Pipeline completo para el modelo tabular (sin imágenes).

    Pasos:
    1. Carga metadata
    2. Split 70/15/15 (ANTES de cualquier transformación)
    3. Imputa nulos en age con la mediana del TRAIN
    4. Estandariza con StandardScaler ajustado solo en TRAIN

    Por qué imputer y scaler se ajustan solo en train:
    Si calculamos la mediana o la media sobre todo el dataset (incluido test),
    el modelo "ve" indirectamente información del test antes de evaluarse.
    Esto se llama data leakage y hace que los resultados sean demasiado optimistas.

    Returns
    -------
    X_tab_train, X_tab_val, X_tab_test : arrays float32
    y_train, y_val, y_test             : arrays one-hot float32
    le                                 : LabelEncoder con las 7 clases
    """
    metadata = load_metadata()

    le = LabelEncoder()
    y_encoded = le.fit_transform(metadata["dx"])
    y_onehot  = to_categorical(y_encoded).astype("float32")

    X = pd.get_dummies(metadata[["age", "sex", "localization"]], drop_first=True)

    (X_train,), (X_val,), (X_test,) = _split_70_15_15(
        [X.values], y_encoded, random_state
    )
    (y_train,), (y_val,), (y_test,) = _split_70_15_15(
        [y_onehot], y_encoded, random_state
    )

    # Imputación (mediana del train)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_val   = imp.transform(X_val)
    X_test  = imp.transform(X_test)

    # Estandarización (fit solo en train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val   = scaler.transform(X_val).astype("float32")
    X_test  = scaler.transform(X_test).astype("float32")

    print(f"Tabular — Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Clases: {le.classes_}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def get_all_splits(random_state=42):
    """
    Pipeline completo para modelos multimodales (imágenes + tabular).

    Pasos:
    1. Carga y alinea ambos CSVs por image_id
    2. Split 70/15/15 (ANTES de cualquier transformación)
    3. Imputa nulos en age con la mediana del TRAIN
    4. Normaliza imágenes dividiendo entre 255

    Returns
    -------
    X_img_train, X_img_val, X_img_test : arrays float32 (n, 28, 28, 3)
    X_tab_train, X_tab_val, X_tab_test : arrays float32 (n, n_features)
    y_train, y_val, y_test             : arrays one-hot float32
    le                                 : LabelEncoder con las 7 clases
    """
    df = load_and_align()

    pixel_cols = [c for c in df.columns
                  if c not in ["image_id", "dx", "age", "sex", "localization"]]

    X_img = df[pixel_cols].values.astype("float32").reshape(-1, 28, 28, 3)
    X_tab_raw = pd.get_dummies(
        df[["age", "sex", "localization"]], columns=["sex", "localization"], drop_first=True
    ).values.astype("float32")

    le = LabelEncoder()
    y_encoded = le.fit_transform(df["dx"].values)
    y_onehot  = to_categorical(y_encoded).astype("float32")

    (X_img_train, X_tab_train), (X_img_val, X_tab_val), (X_img_test, X_tab_test) = \
        _split_70_15_15([X_img, X_tab_raw], y_encoded, random_state)
    (y_train,), (y_val,), (y_test,) = \
        _split_70_15_15([y_onehot], y_encoded, random_state)

    # Imputación tabular (mediana del train)
    imp = SimpleImputer(strategy="median")
    X_tab_train = np.where(np.isinf(X_tab_train), np.nan, X_tab_train)
    X_tab_val   = np.where(np.isinf(X_tab_val),   np.nan, X_tab_val)
    X_tab_test  = np.where(np.isinf(X_tab_test),  np.nan, X_tab_test)
    X_tab_train = imp.fit_transform(X_tab_train).astype("float32")
    X_tab_val   = imp.transform(X_tab_val).astype("float32")
    X_tab_test  = imp.transform(X_tab_test).astype("float32")

    # Normalización per-canal (evita data leakage)
    # Se calcula UNA media y UNA desviación por canal R/G/B, ajustadas SOLO en train.
    # Ventaja frente a /255: centra y escala según la distribución real del dataset,
    # en lugar de asumir que todos los píxeles están uniformemente en [0, 255].
    img_scalers = []
    for c in range(3):
        sc = StandardScaler()
        train_flat = X_img_train[:, :, :, c].reshape(-1, 1)
        sc.fit(train_flat)
        img_scalers.append(sc)

        s_train = X_img_train[:, :, :, c].shape
        s_val   = X_img_val  [:, :, :, c].shape
        s_test  = X_img_test [:, :, :, c].shape

        X_img_train[:, :, :, c] = sc.transform(
            X_img_train[:, :, :, c].reshape(-1, 1)).reshape(s_train)
        X_img_val  [:, :, :, c] = sc.transform(
            X_img_val  [:, :, :, c].reshape(-1, 1)).reshape(s_val)
        X_img_test [:, :, :, c] = sc.transform(
            X_img_test [:, :, :, c].reshape(-1, 1)).reshape(s_test)

    print(f"Imágenes+Tabular — Train: {X_img_train.shape[0]} | Val: {X_img_val.shape[0]} | Test: {X_img_test.shape[0]}")
    print(f"Shape imágenes: {X_img_train.shape[1:]} | Features tabular: {X_tab_train.shape[1]}")
    print(f"Clases: {le.classes_}")
    return (X_img_train, X_img_val, X_img_test,
            X_tab_train, X_tab_val, X_tab_test,
            y_train, y_val, y_test, le)
