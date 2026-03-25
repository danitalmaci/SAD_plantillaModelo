# -*- coding: utf-8 -*-
 
# ======================= PLANTILLA - TEST =======================


"""
Autores: Daniel Talmaci & June Castro
Script para el test de modelos de clasificación.

"""
# -*- coding: utf-8 -*-

import sys
import json
import pickle
import pandas as pd
from colorama import Fore

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score



# ======================= CARGA DE CONFIGURACION =======================

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)



# ======================= PREPROCESADO =======================

def apply_drop_features(data, drop_features):
    if drop_features:
        cols_to_drop = [col for col in drop_features if col in data.columns]
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            print("Columnas eliminadas:", cols_to_drop)
    return data


def apply_missing_values(data, missing_cfg):
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    # Defaults
    num_default = missing_cfg.get("numeric_default", {})
    cat_default = missing_cfg.get("categorical_default", {})
    per_column = missing_cfg.get("per_column", {})

    def fill_column(df, col, strategy_cfg):
        strategy = strategy_cfg.get("strategy", "none")

        if strategy == "none":
            return df

        if strategy == "drop_rows":
            df = df.dropna(subset=[col])
            return df

        if strategy == "mean" and col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(df[col].mean())

        elif strategy == "median" and col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = df[col].fillna(df[col].median())

        elif strategy == "mode":
            moda = df[col].mode()
            if not moda.empty:
                df[col] = df[col].fillna(moda[0])

        elif strategy == "constant":
            value = strategy_cfg.get("value", 0)
            df[col] = df[col].fillna(value)

        return df

    # Primero por columna concreta
    for col, cfg in per_column.items():
        if col in data.columns and data[col].isnull().sum() > 0:
            data = fill_column(data, col, cfg)

    # Luego defaults para el resto
    for col in num_cols:
        if col not in per_column and data[col].isnull().sum() > 0:
            data = fill_column(data, col, num_default)

    for col in cat_cols:
        if col not in per_column and data[col].isnull().sum() > 0:
            data = fill_column(data, col, cat_default)

    return data


def simplify_text(data):
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = data[col].astype(str).str.lower().str.strip()
    return data


def apply_encoding(data):
    cat_cols = data.select_dtypes(include=['object']).columns

    if len(cat_cols) > 0:
        print("Conversión de variables categóricas a numéricas (Label Encoding)")
        for col in cat_cols:
            print(f"Codificando la columna {col}...")
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    return data


def apply_scaling(data, scaling_cfg):
    default_scaling = scaling_cfg.get("default", "none")
    per_column = scaling_cfg.get("per_column", {})

    num_cols = data.select_dtypes(include=['int64', 'float64']).columns

    def get_scaler(name):
        if name == "standard":
            return StandardScaler()
        elif name == "minmax":
            return MinMaxScaler()
        elif name == "maxabs":
            return MaxAbsScaler()
        else:
            return None

    # Por columna concreta
    for col, scaling_type in per_column.items():
        if col in data.columns and col in num_cols:
            scaler = get_scaler(scaling_type)
            if scaler is not None:
                data[[col]] = scaler.fit_transform(data[[col]])
                print(f"Columna {col} escalada con {scaling_type}")

    # Default para el resto
    for col in num_cols:
        if col not in per_column:
            scaler = get_scaler(default_scaling)
            if scaler is not None:
                data[[col]] = scaler.fit_transform(data[[col]])
                print(f"Columna {col} escalada con {default_scaling}")

    return data


def preprocess_test_data(data, config, target_column=None):
    print("\n- Preprocesando datos de test...")

    preprocessing = config.get("preprocessing", {})

    # Quitar target antes de preprocesar X
    y_real = None
    if target_column and target_column in data.columns:
        y_real = data[target_column].copy()
        data = data.drop(columns=[target_column])
        print(f"Columna objetivo '{target_column}' separada correctamente")

    # Drop features
    drop_features = preprocessing.get("drop_features", [])
    data = apply_drop_features(data, drop_features)

    # Missing values
    missing_cfg = preprocessing.get("missing_values", {})
    data = apply_missing_values(data, missing_cfg)

    # Texto simple
    text_process = preprocessing.get("text_process", "none")
    if text_process != "none":
        print("Simplificando texto...")
        data = simplify_text(data)

    # Encoding
    data = apply_encoding(data)

    # Scaling
    scaling_cfg = preprocessing.get("scaling", {"default": "none", "per_column": {}})
    data = apply_scaling(data, scaling_cfg)

    # En TEST no se balancea
    print("No se aplica balanceo en test")

    return data, y_real



# ======================= MODELO =======================

def load_model(model_path):
    """
    Carga el modelo desde el archivo del modelo indicado y lo devuelve.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+f"Modelo cargado con éxito desde {model_path}"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)



# ======================= PROGRAMA PRINCIPAL =======================

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Uso: python test.py config_test.json")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})

    file_path = input_cfg.get("file")
    model_path = input_cfg.get("model_path")
    target_column = input_cfg.get("target", "")

    predictions_file = output_cfg.get("predictions_file", "output/predicciones.csv")

    if not file_path:
        print("Error: falta 'input.file' en el JSON")
        sys.exit(1)

    if not model_path:
        print("Error: falta 'input.model_path' en el JSON")
        sys.exit(1)

    print("\n=== TEST ===")
    print("Fichero de entrada:", file_path)
    print("Modelo:", model_path)
    print("Target:", target_column if target_column else "(no especificado)")

    # Cargar datos
    data = pd.read_csv(file_path)
    print("\nDatos cargados:")
    print(data.head())

    # Preprocesar
    X_test, y_real = preprocess_test_data(data, config, target_column)

    # Cargar modelo
    model = load_model(model_path)

    # Predicción
    print("\n- Realizando predicciones...")
    predictions = model.predict(X_test)

    # Construcción de salida
    results = X_test.copy()
    results["prediccion"] = predictions

    # Si hay valor real, lo añadimos y evaluamos
    if y_real is not None:
        results["valor_real"] = y_real.values

        print("\n=== MÉTRICAS ===")
        try:
            print("F1 macro:", f1_score(y_real, predictions, average="macro"))
            print("\nClassification report:")
            print(classification_report(y_real, predictions))
            print("Matriz de confusión:")
            print(confusion_matrix(y_real, predictions))
        except Exception as e:
            print("No se han podido calcular las métricas:", e)

    # Guardar resultados
    results.to_csv(predictions_file, index=False)
    print(f"\nPredicciones guardadas en: {predictions_file}")
