# -*- coding: utf-8 -*-

# ======================= PLANTILLA - TEST =======================

"""
Autores: Daniel Talmaci & June Castro
Script para el test de modelos de clasificación.
"""

import sys
import json
import pickle
import string
import argparse
import signal
import os
import pandas as pd
from colorama import Fore

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# ======================= CARGA DE CONFIGURACION =======================

def signal_handler(sig, frame): 
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """

    print("\nSaliendo del programa...") 
    sys.exit(0)  

def parse_args(): 
    """
    Función para parsear los argumentos de entrada
    """

    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")

    # Parametros necesarios
    parse.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    # Parametros opcionales
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    
    args = parse.parse_args()
    # Lee los argumentos realmente escritos en la terminal y los guarda en args.

    try:  
        with open(args.json, 'r') as json_file:  # Abre el archivo JSON indicado en modo lectura.
            config = json.load(json_file)  # Carga el contenido del JSON como diccionario Python.

        for key, value in config.items():  # Recorre todas las claves y valores del JSON.
            setattr(args, key, value)  # Añade cada clave del JSON como atributo dentro de args.

    except FileNotFoundError: 
        print(f"Error: No se encontró el archivo {args.json}")  
        sys.exit(1)  

    return args 

# ======================= MODELO =======================

def load_model(model_path):
    """
    Carga el modelo desde el archivo indicado y lo devuelve.
    """
    try:
        with open(model_path, 'rb') as file:
            package = pickle.load(file)
            print(Fore.GREEN + f"Modelo cargado con éxito desde {model_path}" + Fore.RESET)
            return package
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)


# ======================= FUNCIONES AUXILIARES =======================

def get_missing_config(config):
    return config.get("preprocessing", {}).get("missing_values", {})

def get_scaling_config(config):
    return config.get("preprocessing", {}).get("scaling", {})

def get_metrics_average(config):
    return config.get("metrics", {}).get("fscore_average", "macro")

def build_scaler(method):
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "maxabs":
        return MaxAbsScaler()
    if method in ["none", None]:
        return None
    raise ValueError(f"Método de escalado no soportado: {method}")

def select_features(df, config, package):
    """
    Separa las características del conjunto de datos en numéricas, de texto y categóricas.
    """
    try:
        numerical_columns = package.get("numerical_columns", [])
        categorical_columns = package.get("categorical_columns", [])
        text_columns = package.get("text_columns", [])

        numerical_feature = df[[col for col in numerical_columns if col in df.columns]].copy()
        categorical_feature = df[[col for col in categorical_columns if col in df.columns]].copy()
        text_feature = df[[col for col in text_columns if col in df.columns]].copy()

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)
        print("Columnas numéricas:", list(numerical_feature.columns))
        print("Columnas categóricas:", list(categorical_feature.columns))
        print("Columnas de texto:", list(text_feature.columns))

        return numerical_feature, text_feature, categorical_feature

    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


# ======================= PREPROCESADO =======================

def process_missing_values(data, numerical_feature, categorical_feature, config, package):
    """
    Procesa los valores faltantes usando los valores aprendidos en train.
    """
    missing_info = package.get("missing_values_info", {})

    for col in numerical_feature.columns:
        if col in data.columns and data[col].isnull().sum() > 0:
            if col in missing_info:
                strategy = missing_info[col].get("strategy", "none")

                if strategy == "drop_rows":
                    data = data.dropna(subset=[col])
                    print(f"Se eliminan filas con missing en '{col}'")
                elif strategy in ["mean", "median", "mode", "constant"]:
                    fill_value = missing_info[col].get("fill_value", None)
                    data[col] = data[col].fillna(fill_value)
                    print(f"Se imputa en '{col}' el valor aprendido en train: {fill_value}")
                elif strategy == "none":
                    print(f"No se aplica imputación en '{col}'")
            else:
                print(f"No se aplica imputación en '{col}'")

    for col in categorical_feature.columns:
        if col in data.columns and data[col].isnull().sum() > 0:
            if col in missing_info:
                strategy = missing_info[col].get("strategy", "none")

                if strategy == "drop_rows":
                    data = data.dropna(subset=[col])
                    print(f"Se eliminan filas con missing en '{col}'")
                elif strategy in ["mode", "constant", "mean", "median"]:
                    fill_value = missing_info[col].get("fill_value", None)
                    data[col] = data[col].fillna(fill_value)
                    print(f"Se imputa en '{col}' el valor aprendido en train: {fill_value}")
                elif strategy == "none":
                    print(f"No se aplica imputación en '{col}'")
            else:
                print(f"No se aplica imputación en '{col}'")

    return data

def simplify_text(data, text_feature):
    """
    Simplifica el texto: minúsculas, quitar puntuación, tokenizar, eliminar stopwords y stemming.
    """
    print("Simplificando texto...")

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def procesar_texto(texto):
        tokens = word_tokenize(texto)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    for col in text_feature.columns:
        print(f"Procesando columna {col}...")

        data[col] = data[col].fillna("")
        data[col] = data[col].str.lower()
        data[col] = data[col].str.translate(str.maketrans('', '', string.punctuation))
        data[col] = data[col].apply(procesar_texto)

    return data

def cat2num(data, categorical_feature, package):
    """
    Convierte las variables categóricas en numéricas con One-Hot Encoding.
    """
    if categorical_feature.columns.size == 0:
        return data

    print("Conversión de variables categóricas a numéricas (One-Hot Encoding)")

    encoder = package.get("categorical_encoder", None)

    if encoder is None:
        return data

    encoded = encoder.transform(data[categorical_feature.columns])

    encoded_columns = encoder.get_feature_names_out(categorical_feature.columns)
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=data.index)

    data = data.drop(columns=categorical_feature.columns)
    data = pd.concat([data, encoded_df], axis=1)

    print("Nuevas columnas creadas:")
    for col in encoded_columns:
        print(col)

    return data

def reescaler(data, numerical_feature, config, package):
    """
    Reescala las características numéricas usando los scalers aprendidos en train.
    """
    scalers = package.get("scalers", {})

    for col in numerical_feature.columns:
        if col not in data.columns:
            continue

        scaler = scalers.get(col, None)

        if scaler is None:
            print(f"No se escala la columna {col}")
        else:
            data[col] = scaler.transform(data[[col]])
            print(f"Columna {col} escalada con el scaler aprendido en train")

    return data

def process_text(data, text_feature, config, package):
    """
    Procesa las características de texto utilizando TF-IDF o BOW.
    """
    try:
        if text_feature.columns.size > 0:
            text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

            vectorizer = package.get("text_vectorizer", None)
            vectorizer_type = package.get("text_vectorizer_type", "none")

            if vectorizer_type == "tf-idf" and vectorizer is not None:
                tfidf_matrix = vectorizer.transform(text_data)

                text_features_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=data.index
                )

                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)

                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)

            elif vectorizer_type == "bow" and vectorizer is not None:
                bow_matrix = vectorizer.transform(text_data)

                text_features_df = pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=data.index
                )

                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)

                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)

        return data

    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def drop_features(data, config):
    """
    Elimina las columnas especificadas del conjunto de datos.
    """
    try:
        data = data.drop(columns=package.get("drop_features", []), errors="ignore")
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
        print(e)
        sys.exit(1)

def align_features_to_model(X_test, package):
    """
    Alinea las columnas de X_test con las columnas que espera el modelo.
    """
    expected_features = package.get("final_feature_columns", None)

    if expected_features is None:
        print(Fore.YELLOW + "No se han encontrado nombres de columnas esperadas en el modelo. No se alinean columnas." + Fore.RESET)
        return X_test

    for col in expected_features:
        if col not in X_test.columns:
            X_test[col] = 0

    extra_cols = [col for col in X_test.columns if col not in expected_features]
    if extra_cols:
        X_test = X_test.drop(columns=extra_cols)

    X_test = X_test[expected_features]

    print(Fore.GREEN + "Columnas alineadas con el modelo" + Fore.RESET)
    return X_test

def preprocess_test_data(data, config, package, target_column=None):
    """
    Separa la columna objetivo, preprocesa el resto y devuelve X_test e y_real.
    """
    print("\n- Preprocesando datos de test...")

    y_real = None
    if target_column and target_column in data.columns:
        y_real = data[target_column].copy()
        data = data.drop(columns=[target_column])
        print(f"Columna objetivo '{target_column}' separada correctamente")

    numerical_feature, text_feature, categorical_feature = select_features(data, config, package)

    data = process_missing_values(data, numerical_feature, categorical_feature, config, package)
    data = simplify_text(data, text_feature)
    data = cat2num(data, categorical_feature, package)
    data = reescaler(data, numerical_feature, config, package)
    data = process_text(data, text_feature, config, package)
    data = drop_features(data, config)
    data = align_features_to_model(data, package)

    if y_real is not None:
        y_real = y_real.loc[data.index]

    return data, y_real

def calculate_metrics(y_real, predictions, config):
    """
    Calcula las métricas usando la configuración del JSON.
    """
    average_type = get_metrics_average(config)

    if average_type == "micro":
        f1 = f1_score(y_real, predictions, average="micro")
        precision = precision_score(y_real, predictions, average="micro", zero_division=0)
        recall = recall_score(y_real, predictions, average="micro", zero_division=0)
    elif average_type == "macro":
        f1 = f1_score(y_real, predictions, average="macro")
        precision = precision_score(y_real, predictions, average="macro", zero_division=0)
        recall = recall_score(y_real, predictions, average="macro", zero_division=0)
    else:
        f1 = f1_score(y_real, predictions, average="binary")
        precision = precision_score(y_real, predictions, average="binary", zero_division=0)
        recall = recall_score(y_real, predictions, average="binary", zero_division=0)

    return f1, precision, recall


# ======================= PROGRAMA PRINCIPAL =======================

if __name__ == '__main__':
    print("=== Clasificador === ")

    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
   
    config = vars(args) 

    # 3. Acceder a los datos a través de args o config
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

    print("\n=== TEST === ")
    print("Fichero de entrada: ", file_path)
    print("Modelo: ", model_path)
    print("Target: ", target_column if target_column else "(no especificado) ")

    print("\n- Descargando diccionarios... ")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    if not os.path.exists("output"):
        os.makedirs("output")

    data_original = pd.read_csv(file_path)
    print("\nDatos cargados: ")
    print(data_original.head())

    package = load_model(model_path)
    model = package["model"]

    # 4. Pasar config (diccionario) a las funciones de preprocesado
    X_test, y_real = preprocess_test_data(data_original.copy(), config, package, target_column)

    print("\n- Realizando predicciones... ")
    predictions = model.predict(X_test)

    results = data_original.copy()

    if y_real is not None:
        results = results.loc[X_test.index].copy()
        results[target_column + "_REAL"] = y_real.values
    else:
        results = results.loc[X_test.index].copy()

    results[target_column + "_PRED"] = predictions

    print(Fore.GREEN + "Predicción realizada con éxito " + Fore.RESET)

    if y_real is not None:
        print("\n=== MÉTRICAS === ")
        try:
            f1, precision, recall = calculate_metrics(y_real, predictions, config)
            
            print("F1: ", f1)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("\nClassification report: ")
            print(classification_report(y_real, predictions))
            print("Matriz de confusión: ")
            print(confusion_matrix(y_real, predictions))
        except Exception as e:
            print("No se han podido calcular las métricas: ", e)

    results.to_csv(predictions_file, index=False)
    print(Fore.GREEN + f"Predicciones guardadas en: {predictions_file} " + Fore.RESET)
