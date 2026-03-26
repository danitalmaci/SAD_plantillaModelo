# -*- coding: utf-8 -*-  # Indica la codificación del archivo para permitir caracteres como tildes o ñ.

# ======================= PLANTILLA - TRAIN =======================  # Título decorativo de la sección.


"""
Autores: Daniel Talmaci & June Castro
Script para la implementación de los siguientes algoritmos:
        1. kNN
        2. Decision Tree
        3. Random Forest
        4. Naïve Bayes
"""
# Esto es un comentario multilínea o docstring general del archivo.
# Sirve para explicar quién lo ha hecho y qué algoritmos implementa.


# ======================= IMPORTS =======================  # Título decorativo para la sección de importaciones.

import random  # Permite generar números aleatorios.
import sys  # Permite interactuar con el sistema, por ejemplo salir del programa.
import signal  # Permite capturar señales del sistema, como Ctrl+C.
import argparse  # Sirve para leer argumentos que se pasan por terminal.
import pandas as pd  # Librería para trabajar con tablas de datos (DataFrames).
import numpy as np  # Librería para operaciones numéricas y arrays.
import string  # Incluye constantes y utilidades para trabajar con texto.
import pickle  # Permite guardar y cargar objetos Python en archivos.
import time  # Permite medir tiempos y hacer pausas.
import json  # Permite leer y escribir archivos JSON.
import csv  # Permite escribir y leer archivos CSV.
import os  # Permite trabajar con carpetas, rutas y archivos del sistema.
from colorama import Fore  # Permite imprimir texto en colores por terminal.

# Sklearn  # Comentario para indicar que ahora vienen importaciones de scikit-learn.

from sklearn.naive_bayes import GaussianNB  # Importa el clasificador Naive Bayes gaussiano.
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
# Importa métricas para evaluar el modelo: f1, precisión, recall, matriz de confusión e informe de clasificación.

from sklearn.model_selection import train_test_split, GridSearchCV
# train_test_split divide datos en train y test/dev.
# GridSearchCV prueba varias combinaciones de hiperparámetros con validación cruzada.

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
# Escaladores para normalizar o reescalar variables numéricas.

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# TfidfVectorizer convierte texto a números usando TF-IDF.
# CountVectorizer convierte texto a números contando palabras (BOW).

from sklearn.neighbors import KNeighborsClassifier  # Clasificador kNN.
from sklearn.tree import DecisionTreeClassifier  # Clasificador árbol de decisión.
from sklearn.ensemble import RandomForestClassifier  # Clasificador Random Forest.
from sklearn.preprocessing import OneHotEncoder  # Codifica variables categóricas como variables numéricas binarias.

# Nltk  # Comentario para indicar que ahora vienen importaciones de NLTK.

import nltk  # Librería de procesamiento de lenguaje natural.
from nltk.corpus import stopwords  # Lista de palabras vacías, como "the", "is", etc.
from nltk.stem import PorterStemmer  # Herramienta para hacer stemming.
from nltk.tokenize import word_tokenize  # Función para dividir texto en tokens/palabras.

# Imblearn  # Comentario para indicar que ahora vienen importaciones de balanceo.

from imblearn.under_sampling import RandomUnderSampler  # Para hacer undersampling.
from imblearn.over_sampling import RandomOverSampler  # Para hacer oversampling.
from tqdm import tqdm  # Librería para mostrar barras de progreso en terminal.

# ======================= PROGRAMA =======================  # Separador decorativo principal.


# ----------------- Funciones auxiliares -----------------  # Separador de funciones auxiliares.

def signal_handler(sig, frame):  # Define una función que manejará la señal Ctrl+C.
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    # Docstring que explica qué hace la función y sus parámetros.

    print("\nSaliendo del programa...")  # Muestra un mensaje al usuario indicando que el programa termina.
    sys.exit(0)  # Cierra el programa con código 0, que normalmente indica salida correcta.

def parse_args():  # Define una función para leer argumentos de la terminal.
    """
    Función para parsear los argumentos de entrada
    """
    # Docstring explicativa.

    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    # Crea un parser de argumentos con una descripción del programa.

    # Parametros necesarios
    parse.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    # Añade un argumento obligatorio llamado -j o --json para indicar el archivo de configuración.

    # Parametros opcionales
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    # Añade un argumento opcional para indicar la métrica o estimador principal.

    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    # Añade un argumento opcional para indicar cuántos núcleos usar.

    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    # Añade un argumento opcional tipo bandera.
    # Si se usa -v, verbose pasa a valer True.

    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Añade un argumento opcional tipo bandera para modo debug.

    args = parse.parse_args()
    # Lee los argumentos realmente escritos en la terminal y los guarda en args.

    try:  # Empieza un bloque de control de errores.
        with open(args.json, 'r') as json_file:  # Abre el archivo JSON indicado en modo lectura.
            config = json.load(json_file)  # Carga el contenido del JSON como diccionario Python.

        for key, value in config.items():  # Recorre todas las claves y valores del JSON.
            setattr(args, key, value)  # Añade cada clave del JSON como atributo dentro de args.

        if not hasattr(args, "file") and hasattr(args, "data_file"):
            # Comprueba si args no tiene atributo file pero sí tiene data_file.
            args.file = args.data_file
            # Si pasa eso, copia data_file en file para unificar nombres.

    except FileNotFoundError:  # Si no se encuentra el archivo JSON...
        print(f"Error: No se encontró el archivo {args.json}")  # Muestra mensaje de error.
        sys.exit(1)  # Sale del programa con error.

    return args  # Devuelve todos los argumentos ya procesados.

def load_data(file):  # Define una función para cargar un CSV.
    """
    Función para cargar los datos de un fichero csv
    Param
        file: Fichero csv
    Returns:
        Datos del fichero
    """
    # Docstring explicativa de la función.

    try:  # Empieza un bloque de control de errores.
        data = pd.read_csv(file, encoding='utf-8')
        # Lee el archivo CSV con pandas usando codificación UTF-8.

        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        # Imprime un mensaje en verde indicando que la carga fue correcta.

        return data
        # Devuelve el DataFrame leído.

    except Exception as e:  # Si ocurre cualquier error al leer...
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        # Imprime un mensaje en rojo.

        print(e)  # Imprime el error exacto.
        sys.exit(1)  # Termina el programa con error.

# ---------- Funciones para cálculo de Métricas ----------  # Separador decorativo.

def calculate_fscore(y_true, y_pred):  # Función para calcular el F1-score.
    """
    Función para calcular el F-Score del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee del JSON el tipo de media que se quiere usar: micro, macro o none.

    if average_type == "micro":  # Si se quiere media micro...
        return f1_score(y_true, y_pred, average='micro')
        # Devuelve el F1 micro.

    elif average_type == "macro":  # Si se quiere media macro...
        return f1_score(y_true, y_pred, average='macro')
        # Devuelve el F1 macro.

    elif average_type == "none":  # Si no se ha configurado micro ni macro...
        return f1_score(y_true, y_pred)
        # Devuelve el F1 por defecto de sklearn.

    else:  # Si el valor del JSON no es válido...
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
        # Lanza un error indicando el valor incorrecto.
    
def calculate_precision(y_true, y_pred):  # Función para calcular precision.
    """
    Función para calcular la precisión del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee el tipo de media configurado.

    if average_type == "micro":  # Si es micro...
        return precision_score(y_true, y_pred, average='micro', zero_division=0)
        # Devuelve precision micro. Si hay división entre 0, devuelve 0.

    elif average_type == "macro":  # Si es macro...
        return precision_score(y_true, y_pred, average='macro', zero_division=0)
        # Devuelve precision macro.

    elif average_type == "none":  # Si es none...
        return precision_score(y_true, y_pred, zero_division=0)
        # Devuelve precision normal.

    else:  # Si el valor no es válido...
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
        # Lanza error.


def calculate_recall(y_true, y_pred):  # Función para calcular recall.
    """
    Función para calcular el recall del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee el tipo de media configurado.

    if average_type == "micro":  # Si es micro...
        return recall_score(y_true, y_pred, average='micro', zero_division=0)
        # Devuelve recall micro.

    elif average_type == "macro":  # Si es macro...
        return recall_score(y_true, y_pred, average='macro', zero_division=0)
        # Devuelve recall macro.

    elif average_type == "none":  # Si es none...
        return recall_score(y_true, y_pred, zero_division=0)
        # Devuelve recall normal.

    else:  # Si el valor no es válido...
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
        # Lanza error.

def calculate_confusion_matrix(y_true, y_pred):  # Función para matriz de confusión.
    """
    Función para calcular la matriz de confusión

    Returns:
        Matriz de confusión
    """
    return confusion_matrix(y_true, y_pred)
    # Devuelve la matriz de confusión comparando valores reales y predichos.

def calculate_classification_report(y_true, y_pred):  # Función para classification report.
    """
    Función para calcular el informe de clasificación

    Returns:
        Informe de clasificación
    """
    return classification_report(y_true, y_pred)
    # Devuelve el informe completo de clasificación.
    
def get_scoring_metrics():  # Función para construir el diccionario de métricas para GridSearchCV.
    """
    Devuelve las métricas de scoring para GridSearchCV según la configuración del JSON.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee del JSON el tipo de media a utilizar.

    if average_type == "micro":  # Si se usa micro...
        return {
            "score": args.estimator,  # Métrica principal para escoger el mejor modelo.
            "precision": "precision_micro",  # Métrica secundaria precision micro.
            "recall": "recall_micro"  # Métrica secundaria recall micro.
        }

    elif average_type == "macro":  # Si se usa macro...
        return {
            "score": args.estimator,  # Métrica principal.
            "precision": "precision_macro",  # Precision macro.
            "recall": "recall_macro"  # Recall macro.
        }

    elif average_type == "none":  # Si no se usa ni micro ni macro...
        return {
            "score": args.estimator,  # Métrica principal.
            "precision": "precision",  # Precision normal.
            "recall": "recall"  # Recall normal.
        }

    else:  # Si el valor no es válido...
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
        # Lanza error.

# -------------- Funciones de Preprocesado --------------  # Separador decorativo.

def select_features(df):  # Función para separar columnas según tipo.
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.
    """
    try:  # Empieza bloque de control de errores.
        numerical_feature = df.select_dtypes(include=['int64', 'float64']).copy()
        # Selecciona columnas numéricas enteras o decimales y hace una copia.

        categorical_feature = df.select_dtypes(include='object').copy()
        # Selecciona todas las columnas de tipo texto/object y hace una copia.

        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        # De esas columnas de texto, considera categóricas solo las que tengan pocos valores únicos.

        text_feature = df.select_dtypes(include='object').drop(columns=categorical_feature.columns, errors='ignore').copy()
        # Las columnas object que no sean categóricas se consideran de texto libre.

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        # Muestra mensaje de éxito.

        if args.debug:  # Si el modo debug está activado...
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            # Imprime las columnas numéricas.

            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            # Imprime las columnas de texto.

            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
            # Imprime las columnas categóricas.

        return numerical_feature, text_feature, categorical_feature
        # Devuelve los tres grupos de columnas.

    except Exception as e:  # Si ocurre cualquier error...
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        # Muestra mensaje de error.

        print(e)  # Imprime el error concreto.
        sys.exit(1)  # Termina el programa.

def get_missing_config():  # Función auxiliar para leer la config de missing values.
    """
    Devuelve la configuración de tratamiento de valores perdidos.
    """
    return args.preprocessing.get("missing_values", {})
    # Devuelve el bloque missing_values del JSON, o un diccionario vacío si no existe.

def get_scaling_config():  # Función auxiliar para leer la config de escalado.
    """
    Devuelve la configuración de escalado.
    """
    return args.preprocessing.get("scaling", {})
    # Devuelve el bloque scaling del JSON, o vacío si no existe.

def get_balancing_config():  # Función auxiliar para leer la config de balanceo.
    """
    Devuelve la configuración de balanceo.
    """
    return args.preprocessing.get("balancing", {})
    # Devuelve el bloque balancing del JSON, o vacío si no existe.

def build_scaler(method):  # Función que crea el escalador adecuado según el método.
    """
    Función que crea el scaler correspondiente al método indicado.
    """
    if method == "standard":  # Si se pide escalado estándar...
        return StandardScaler()
        # Devuelve un StandardScaler.

    if method == "minmax":  # Si se pide min-max...
        return MinMaxScaler()
        # Devuelve un MinMaxScaler.

    if method == "maxabs":  # Si se pide maxabs...
        return MaxAbsScaler()
        # Devuelve un MaxAbsScaler.

    if method in ["none", None]:  # Si no se quiere escalado...
        return None
        # Devuelve None.

    raise ValueError(f"Método de escalado no soportado: {method}")
    # Si el método no existe, lanza error.

def check_imbalance():  # Función para comprobar si el dataset está desbalanceado.
    """
    Comprueba si el dataset completo está desbalanceado según el umbral del JSON.
    """
    global data  # Indica que se va a usar la variable global data.

    balancing_cfg = get_balancing_config()
    # Obtiene la configuración de balanceo.

    method = balancing_cfg.get("method", "none")
    # Lee el método de balanceo. Si no existe, usa "none".

    if method == "none":  # Si el método es none...
        print("No se comprobará balanceo porque el método configurado es 'none'")
        # Informa de que no se va a balancear.

        return False
        # Devuelve False, es decir, no se considera necesario balancear.

    threshold = balancing_cfg.get("imbalance_threshold", 0.8)
    # Obtiene el umbral de desbalanceo. Por defecto 0.8.

    class_counts = data[args.prediction].value_counts()
    # Cuenta cuántas muestras hay de cada clase en la variable objetivo.

    if len(class_counts) < 2:  # Si solo hay una clase...
        print("No se aplica balanceo porque solo hay una clase")
        # Muestra mensaje.

        return False
        # No tiene sentido balancear.

    majority_count = class_counts.max()
    # Obtiene el número de muestras de la clase mayoritaria.

    minority_count = class_counts.min()
    # Obtiene el número de muestras de la clase minoritaria.

    ratio = minority_count / majority_count
    # Calcula la relación entre clase minoritaria y mayoritaria.

    print("Distribución de clases antes del split:")
    # Imprime título informativo.

    print(class_counts)
    # Imprime el conteo por clase.

    print(f"Ratio minoritaria/mayoritaria: {ratio:.4f}")
    # Imprime el ratio con 4 decimales.

    if ratio < threshold:  # Si la relación es menor que el umbral...
        print("El dataset se considera desbalanceado")
        # Indica que sí hay desbalanceo.

        return True
        # Devuelve True.

    print("El dataset no se considera desbalanceado")
    # Si no, informa que no está desbalanceado.

    return False
    # Devuelve False.

def over_under_sampling(x_train, y_train):  # Función para balancear solo train.
    """
    Realiza oversampling o undersampling SOLO sobre train.
    """
    balancing_cfg = get_balancing_config()
    # Lee la configuración de balanceo.

    method = balancing_cfg.get("method", "none")
    # Obtiene el método.

    random_state = balancing_cfg.get("random_state", 42)
    # Obtiene semilla aleatoria para reproducibilidad.

    if method == "none":  # Si no se quiere balanceo...
        print("No se aplica balanceo en train porque el método configurado es 'none'")
        # Informa de ello.

        return x_train, y_train
        # Devuelve train sin cambios.

    if method == "oversampling":  # Si se quiere oversampling...
        sampler = RandomOverSampler(random_state=random_state)
        # Crea el objeto para duplicar ejemplos de la clase minoritaria.

        print("Aplicando Oversampling SOLO en train...")
        # Muestra mensaje.

    elif method == "undersampling":  # Si se quiere undersampling...
        sampler = RandomUnderSampler(random_state=random_state)
        # Crea el objeto para reducir ejemplos de la clase mayoritaria.

        print("Aplicando Undersampling SOLO en train...")
        # Muestra mensaje.

    else:  # Si el método no es válido...
        raise ValueError(f"Método de balanceo no soportado: {method}")
        # Lanza error.

    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)
    # Aplica el balanceo y devuelve nuevos datos de train balanceados.

    print("Balanceo aplicado correctamente sobre train")
    # Informa del éxito.

    print("Distribución de clases en y_train después del balanceo:")
    # Título informativo.

    print(pd.Series(y_train_resampled).value_counts())
    # Muestra el nuevo reparto de clases tras balancear.

    return x_train_resampled, y_train_resampled
    # Devuelve los nuevos datos de train.

def process_missing_values(x_train, x_dev, y_train, y_dev, numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en train y dev usando estadísticas calculadas SOLO con train.
    """
    missing_cfg = get_missing_config()
    # Lee la configuración de missing values.

    per_column = missing_cfg.get("per_column", {})
    # Obtiene la configuración específica por columna.

    all_columns = list(numerical_feature.columns) + list(categorical_feature.columns)
    # Junta en una sola lista las columnas numéricas y categóricas.

    for col in all_columns:  # Recorre cada columna.
        if col not in per_column:  # Si esa columna no tiene configuración específica...
            continue
            # La salta.

        strategy_cfg = per_column[col]
        # Guarda la configuración concreta de esa columna.

        strategy = strategy_cfg.get("strategy", "none")
        # Obtiene la estrategia a usar.

        is_numeric = col in numerical_feature.columns
        # Comprueba si la columna es numérica.

        if strategy == "drop_rows":  # Si la estrategia es eliminar filas con missing...
            train_mask = x_train[col].notna()
            # Crea una máscara booleana con las filas de train que no son NaN.

            dev_mask = x_dev[col].notna()
            # Crea una máscara igual para dev.

            x_train = x_train.loc[train_mask].copy()
            # Filtra x_train dejando solo filas sin missing en esa columna.

            y_train = y_train.loc[train_mask].copy()
            # Filtra y_train de forma consistente.

            x_dev = x_dev.loc[dev_mask].copy()
            # Filtra x_dev.

            y_dev = y_dev.loc[dev_mask].copy()
            # Filtra y_dev.

            print(f"Se eliminan filas con missing en '{col}'")
            # Informa de lo hecho.

        elif strategy == "mean" and is_numeric:  # Si la estrategia es media y la columna es numérica...
            fill_value = x_train[col].mean()
            # Calcula la media SOLO con train.

            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena los NaN de train con la media.

            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena los NaN de dev con la media calculada en train.

            print(f"Se imputa la media en '{col}'")
            # Informa.

        elif strategy == "median" and is_numeric:  # Si la estrategia es mediana...
            fill_value = x_train[col].median()
            # Calcula la mediana en train.

            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena train.

            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena dev con el mismo valor.

            print(f"Se imputa la mediana en '{col}'")
            # Informa.

        elif strategy == "mode":  # Si la estrategia es moda...
            if not x_train[col].mode().empty:
                # Comprueba que la moda exista.

                fill_value = x_train[col].mode()[0]
                # Toma el valor más frecuente.

                x_train[col] = x_train[col].fillna(fill_value)
                # Rellena train.

                x_dev[col] = x_dev[col].fillna(fill_value)
                # Rellena dev con el mismo valor.

                print(f"Se imputa la moda en '{col}'")
                # Informa.

        elif strategy == "constant":  # Si la estrategia es valor constante...
            fill_value = strategy_cfg.get("value", 0 if is_numeric else "desconocido")
            # Toma el valor definido en JSON.
            # Si no hay valor, usa 0 si es numérica o "desconocido" si no lo es.

            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena train.

            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena dev.

            print(f"Se imputa un valor constante en '{col}': {fill_value}")
            # Informa del valor usado.

        elif strategy == "none":  # Si no se quiere hacer nada...
            print(f"No se aplica imputación en '{col}'")
            # Informa.

        else:  # Si la estrategia no es válida...
            raise ValueError(f"Estrategia de missing no válida para '{col}': {strategy}")
            # Lanza error.

    return x_train, x_dev, y_train, y_dev
    # Devuelve train y dev ya tratados.

def simplify_text(x_train, x_dev, text_feature):  # Función para simplificar texto.
    """
    Simplifica el texto en train y dev.
    """
    print("Simplificando texto...")
    # Mensaje informativo.

    stop_words = set(stopwords.words('english'))
    # Carga las stopwords en inglés y las mete en un conjunto.

    stemmer = PorterStemmer()
    # Crea el objeto para hacer stemming.

    def procesar_texto(texto):  # Función interna para procesar una cadena de texto.
        tokens = word_tokenize(texto)
        # Divide el texto en palabras/tokens.

        tokens = [t for t in tokens if t not in stop_words]
        # Elimina las stopwords.

        tokens = [stemmer.stem(t) for t in tokens]
        # Aplica stemming a cada token.

        return " ".join(tokens)
        # Vuelve a unir los tokens en una cadena.

    for col in text_feature.columns:  # Recorre cada columna de texto.
        print(f"Procesando columna {col}...")
        # Informa de qué columna está tratando.

        x_train[col] = x_train[col].fillna("")
        # Sustituye NaN por cadena vacía en train.

        x_dev[col] = x_dev[col].fillna("")
        # Sustituye NaN por cadena vacía en dev.

        x_train[col] = x_train[col].str.lower()
        # Convierte a minúsculas el texto de train.

        x_dev[col] = x_dev[col].str.lower()
        # Convierte a minúsculas el texto de dev.

        x_train[col] = x_train[col].str.translate(str.maketrans('', '', string.punctuation))
        # Elimina signos de puntuación en train.

        x_dev[col] = x_dev[col].str.translate(str.maketrans('', '', string.punctuation))
        # Elimina signos de puntuación en dev.

        x_train[col] = x_train[col].apply(procesar_texto)
        # Aplica la función de tokenizar, quitar stopwords y stemmizar a train.

        x_dev[col] = x_dev[col].apply(procesar_texto)
        # Hace lo mismo en dev.

    return x_train, x_dev
    # Devuelve train y dev con texto simplificado.

def cat2num(x_train, x_dev, categorical_feature):  # Función para convertir categóricas a numéricas.
    """
    Convierte las características categóricas en características numéricas utilizando One-Hot Encoding.
    """
    if categorical_feature.columns.size == 0:  # Si no hay columnas categóricas...
        return x_train, x_dev
        # Devuelve los datos tal cual.

    print("Conversión de variables categóricas a numéricas (One-Hot Encoding)")
    # Muestra mensaje.

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    # Crea el encoder.
    # sparse=False hace que devuelva array normal en vez de matriz dispersa.
    # handle_unknown="ignore" ignora categorías nuevas en dev.

    encoded_train = encoder.fit_transform(x_train[categorical_feature.columns])
    # Ajusta el encoder con train y transforma esas columnas en variables dummy.

    encoded_dev = encoder.transform(x_dev[categorical_feature.columns])
    # Transforma dev usando el encoder ya ajustado con train.

    encoded_columns = encoder.get_feature_names_out(categorical_feature.columns)
    # Obtiene los nombres de las nuevas columnas generadas.

    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_columns, index=x_train.index)
    # Convierte el array codificado de train en DataFrame.

    encoded_dev_df = pd.DataFrame(encoded_dev, columns=encoded_columns, index=x_dev.index)
    # Convierte el array codificado de dev en DataFrame.

    x_train = x_train.drop(columns=categorical_feature.columns)
    # Elimina de x_train las columnas categóricas originales.

    x_dev = x_dev.drop(columns=categorical_feature.columns)
    # Elimina de x_dev las columnas categóricas originales.

    x_train = pd.concat([x_train, encoded_train_df], axis=1)
    # Une las nuevas columnas binarias al x_train original.

    x_dev = pd.concat([x_dev, encoded_dev_df], axis=1)
    # Une las nuevas columnas binarias al x_dev original.

    print("Nuevas columnas creadas:")
    # Muestra título informativo.

    for col in encoded_columns:  # Recorre cada nueva columna creada.
        print(col)  # La imprime.

    return x_train, x_dev
    # Devuelve train y dev ya transformados.

def reescaler(x_train, x_dev, numerical_feature):  # Función para escalar variables numéricas.
    """
    Reescala las características numéricas usando la configuración del JSON.
    El scaler se ajusta con train y se aplica a train y dev.
    """
    scaling_cfg = get_scaling_config()
    # Obtiene configuración de escalado.

    default_method = scaling_cfg.get("default", "none")
    # Método por defecto para todas las columnas.

    per_column = scaling_cfg.get("per_column", {})
    # Métodos específicos para columnas concretas.

    for col in numerical_feature.columns:  # Recorre todas las columnas numéricas.
        if col not in x_train.columns:  # Si esa columna ya no está...
            continue
            # La salta.

        method = per_column.get(col, default_method)
        # Toma el método específico de la columna o el por defecto.

        scaler = build_scaler(method)
        # Construye el escalador correspondiente.

        if scaler is None:  # Si no hay escalador...
            print(f"No se escala la columna {col}")
            # Informa de que no se toca.

        else:  # Si sí hay escalador...
            x_train[col] = scaler.fit_transform(x_train[[col]])
            # Ajusta el scaler con train y transforma la columna en train.

            x_dev[col] = scaler.transform(x_dev[[col]])
            # Usa el mismo scaler para transformar dev.

            print(f"Columna {col} escalada con {method}")
            # Informa del método usado.

    return x_train, x_dev
    # Devuelve train y dev escalados.

def process_text(x_train, x_dev, text_feature):  # Función para vectorizar texto.
    """
    Procesa las características de texto utilizando TF-IDF o BOW.
    El vectorizador se ajusta con train y se aplica a train y dev.
    """
    try:  # Empieza bloque de errores.
        if text_feature.columns.size > 0:  # Si hay columnas de texto...
            text_train = x_train[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            # Une todas las columnas de texto de cada fila en un solo texto para train.

            text_dev = x_dev[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            # Hace lo mismo en dev.

            if args.preprocessing["text_process"] == "tf-idf":  # Si se quiere TF-IDF...
                tfidf_vectorizer = TfidfVectorizer()
                # Crea el vectorizador TF-IDF.

                tfidf_train = tfidf_vectorizer.fit_transform(text_train)
                # Ajusta el vectorizador con train y transforma train.

                tfidf_dev = tfidf_vectorizer.transform(text_dev)
                # Transforma dev con el vectorizador aprendido en train.

                train_text_df = pd.DataFrame(
                    tfidf_train.toarray(),  # Convierte la matriz dispersa en array normal.
                    columns=tfidf_vectorizer.get_feature_names_out(),  # Usa las palabras como nombres de columnas.
                    index=x_train.index  # Mantiene los índices originales.
                )

                dev_text_df = pd.DataFrame(
                    tfidf_dev.toarray(),  # Convierte la matriz de dev a array.
                    columns=tfidf_vectorizer.get_feature_names_out(),  # Mismos nombres de columnas.
                    index=x_dev.index  # Mismos índices.
                )

                x_train = pd.concat([x_train, train_text_df], axis=1)
                # Añade al x_train las nuevas columnas numéricas del texto.

                x_dev = pd.concat([x_dev, dev_text_df], axis=1)
                # Añade al x_dev las nuevas columnas.

                x_train.drop(text_feature.columns, axis=1, inplace=True)
                # Elimina las columnas de texto originales de train.

                x_dev.drop(text_feature.columns, axis=1, inplace=True)
                # Elimina las columnas de texto originales de dev.

                print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
                # Informa del éxito.

            elif args.preprocessing["text_process"] == "bow":  # Si se quiere bolsa de palabras...
                bow_vectorizer = CountVectorizer()
                # Crea el vectorizador BOW.

                bow_train = bow_vectorizer.fit_transform(text_train)
                # Ajusta y transforma train.

                bow_dev = bow_vectorizer.transform(text_dev)
                # Transforma dev.

                train_text_df = pd.DataFrame(
                    bow_train.toarray(),  # Convierte a array.
                    columns=bow_vectorizer.get_feature_names_out(),  # Nombres de palabras.
                    index=x_train.index  # Índices.
                )

                dev_text_df = pd.DataFrame(
                    bow_dev.toarray(),  # Convierte a array.
                    columns=bow_vectorizer.get_feature_names_out(),  # Nombres.
                    index=x_dev.index  # Índices.
                )

                x_train = pd.concat([x_train, train_text_df], axis=1)
                # Añade columnas BOW a train.

                x_dev = pd.concat([x_dev, dev_text_df], axis=1)
                # Añade columnas BOW a dev.

                x_train.drop(text_feature.columns, axis=1, inplace=True)
                # Elimina texto original de train.

                x_dev.drop(text_feature.columns, axis=1, inplace=True)
                # Elimina texto original de dev.

                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
                # Informa del éxito.

            else:  # Si text_process no es tf-idf ni bow...
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
                # Informa de que no se procesará el texto.
        else:  # Si no hay columnas de texto...
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
            # Informa de que no hay nada que procesar.

        return x_train, x_dev
        # Devuelve train y dev.

    except Exception as e:  # Si ocurre cualquier error...
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        # Mensaje de error.

        print(e)  # Imprime el error exacto.
        sys.exit(1)  # Cierra el programa.

def drop_features(x_train, x_dev):  # Función para eliminar columnas.
    """
    Elimina las columnas especificadas del conjunto de datos.
    """
    try:  # Empieza bloque de control de errores.
        features_to_drop = args.preprocessing.get("drop_features", [])
        # Lee del JSON qué columnas hay que eliminar.

        x_train = x_train.drop(columns=features_to_drop, errors="ignore")
        # Elimina esas columnas de train. Si alguna no existe, la ignora.

        x_dev = x_dev.drop(columns=features_to_drop, errors="ignore")
        # Elimina esas columnas de dev.

        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
        # Informa del éxito.

        return x_train, x_dev
        # Devuelve train y dev.

    except Exception as e:  # Si ocurre un error...
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        # Mensaje de error.

        print(e)  # Imprime el error concreto.
        sys.exit(1)  # Sale del programa.

def preprocesar_datos(x_train, x_dev, y_train, y_dev):  # Función general que junta todo el preprocesado.
    """
    Preprocesa train y dev después del split.
    """
    numerical_feature, text_feature, categorical_feature = select_features(x_train)
    # Separa las columnas de x_train en numéricas, texto y categóricas.

    x_train, x_dev, y_train, y_dev = process_missing_values(
        x_train, x_dev, y_train, y_dev, numerical_feature, categorical_feature
    )
    # Trata los valores perdidos.

    x_train, x_dev = simplify_text(x_train, x_dev, text_feature)
    # Simplifica el texto.

    x_train, x_dev = cat2num(x_train, x_dev, categorical_feature)
    # Convierte categóricas a numéricas.

    x_train, x_dev = reescaler(x_train, x_dev, numerical_feature)
    # Escala las columnas numéricas.

    x_train, x_dev = process_text(x_train, x_dev, text_feature)
    # Convierte texto a variables numéricas con TF-IDF o BOW.

    x_train, x_dev = drop_features(x_train, x_dev)
    # Elimina columnas indicadas en el JSON.

    return x_train, x_dev, y_train, y_dev
    # Devuelve todo ya preprocesado.

# Funciones para entrenar un modelo  # Comentario de sección.

def divide_data():  # Función para dividir los datos.
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    global data  # Usa la variable global data.

    try:  # Bloque de control de errores.
        X = data.drop(columns=[args.prediction])
        # Crea X quitando la columna objetivo.

        y = data[args.prediction]
        # Crea y con la columna objetivo.

        x_train, x_dev, y_train, y_dev = train_test_split(
            X,  # Variables de entrada.
            y,  # Variable objetivo.
            test_size=args.split["test_size"],  # Proporción para dev/test.
            random_state=args.split["random_state"],  # Semilla para reproducibilidad.
            stratify=y  # Mantiene la proporción de clases en train y dev.
        )

        print(Fore.GREEN + "Datos divididos con éxito" + Fore.RESET)
        # Informa de que la división ha salido bien.

        if args.debug:  # Si debug está activado...
            print(Fore.MAGENTA + "> Tamaño x_train:" + Fore.RESET, x_train.shape)
            # Imprime tamaño de x_train.

            print(Fore.MAGENTA + "> Tamaño x_dev:" + Fore.RESET, x_dev.shape)
            # Imprime tamaño de x_dev.

            print(Fore.MAGENTA + "> Tamaño y_train:" + Fore.RESET, y_train.shape)
            # Imprime tamaño de y_train.

            print(Fore.MAGENTA + "> Tamaño y_dev:" + Fore.RESET, y_dev.shape)
            # Imprime tamaño de y_dev.

        return x_train, x_dev, y_train, y_dev
        # Devuelve los cuatro conjuntos.

    except Exception as e:  # Si ocurre error...
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        # Mensaje de error.

        print(e)  # Muestra el error concreto.
        sys.exit(1)  # Termina el programa.

def save_model(gs, algorithm_name):  # Función para guardar el modelo y resultados.
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.
    """
    try:  # Bloque de control de errores.
        with open('output/modelo.pkl', 'wb') as file:
            # Abre el archivo donde se guardará el modelo en binario escritura.

            pickle.dump(gs, file)
            # Guarda el objeto gs en ese archivo.

            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
            # Informa del éxito.

        with open('output/modelo.csv', 'w', newline='') as file:
            # Abre un CSV para guardar el resumen de resultados.

            writer = csv.writer(file)
            # Crea un escritor CSV.

            writer.writerow(['Algoritmo', 'Params', 'Score', 'Precision', 'Recall'])
            # Escribe la fila de cabecera.

            for params, score, precision, recall in zip(
                gs.cv_results_['params'],  # Parámetros probados.
                gs.cv_results_['mean_test_score'],  # Score medio.
                gs.cv_results_['mean_test_precision'],  # Precision media.
                gs.cv_results_['mean_test_recall']  # Recall medio.
            ):
                writer.writerow([algorithm_name, params, score, precision, recall])
                # Escribe una fila por cada combinación de parámetros.

    except Exception as e:  # Si ocurre error...
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        # Muestra mensaje de error.

        print(e)  # Imprime el error.

def mostrar_resultados(gs, x_dev, y_dev):  # Función para mostrar resultados por terminal.
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score del clasificador en el conjunto de desarrollo.
    - Precision del clasificador en el conjunto de desarrollo.
    - Recall del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:  # Solo muestra resultados detallados si verbose está activado.
        y_pred = gs.predict(x_dev)
        # Hace predicciones sobre el conjunto de desarrollo.

        average_type = args.metrics.get("fscore_average", "none")
        # Lee el tipo de media configurado.

        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        # Imprime la mejor combinación de hiperparámetros.

        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        # Imprime el mejor score obtenido en GridSearchCV.

        if average_type == "micro":  # Si se usa micro...
            print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1 micro.

            print(Fore.MAGENTA+"> Precision micro:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision micro.

            print(Fore.MAGENTA+"> Recall micro:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall micro.

        elif average_type == "macro":  # Si se usa macro...
            print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1 macro.

            print(Fore.MAGENTA+"> Precision macro:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision macro.

            print(Fore.MAGENTA+"> Recall macro:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall macro.

        elif average_type == "none":  # Si se usa configuración normal...
            print(Fore.MAGENTA+"> F1-score:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1.

            print(Fore.MAGENTA+"> Precision:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision.

            print(Fore.MAGENTA+"> Recall:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall.

        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, y_pred))
        # Imprime el classification report completo.

        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, y_pred))
        # Imprime la matriz de confusión.

def kNN():  # Función que ejecuta el entrenamiento con kNN.
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.

    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    if is_imbalanced:  # Si hay desbalanceo...
        x_train, y_train = over_under_sampling(x_train, y_train)
        # Aplica balanceo solo al train.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev)
    # Aplica todo el preprocesado.

    if args.debug:  # Si modo debug está activado...
        try:
            train_debug = x_train.copy()
            # Hace copia de x_train.

            train_debug[args.prediction] = y_train.values
            # Añade la columna objetivo al train procesado.

            dev_debug = x_dev.copy()
            # Hace copia de x_dev.

            dev_debug[args.prediction] = y_dev.values
            # Añade la columna objetivo al dev procesado.

            train_debug.to_csv('output/train-processed.csv', index=False)
            # Guarda el train procesado.

            dev_debug.to_csv('output/dev-processed.csv', index=False)
            # Guarda el dev procesado.

            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
            # Informa del éxito.

        except Exception as e:  # Si hay error...
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            # Mensaje de error.

            print(e)  # Imprime error.

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        # Crea una barra de progreso visual.

        gs = GridSearchCV(
            KNeighborsClassifier(),  # Modelo base kNN.
            args.kNN,  # Hiperparámetros a probar, leídos del JSON.
            cv=5,  # Validación cruzada con 5 particiones.
            n_jobs=args.cpu,  # Número de CPUs a usar.
            scoring=get_scoring_metrics(),  # Métricas de evaluación.
            refit="score"  # Reentrena usando como métrica principal "score".
        )

        start_time = time.time()
        # Guarda el instante inicial.

        gs.fit(x_train, y_train)
        # Entrena el GridSearchCV.

        end_time = time.time()
        # Guarda el instante final.

        for i in range(100):  # Bucle artificial para animar la barra.
            time.sleep(random.uniform(0.06, 0.15))
            # Espera un tiempo aleatorio pequeño.

            pbar.update(random.random() * 2)
            # Actualiza la barra de progreso en una cantidad aleatoria.

        pbar.n = 100
        # Fuerza el progreso al 100%.

        pbar.last_print_n = 100
        # Ajusta la última impresión de la barra.

        pbar.update(0)
        # Refresca la barra sin sumar nada.

    execution_time = end_time - start_time
    # Calcula el tiempo total de entrenamiento.

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    # Imprime el tiempo.

    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra resultados si verbose está activo.

    save_model(gs,"kNN")
    # Guarda el modelo y los resultados.

def decision_tree():  # Función para entrenar árbol de decisión.
    """
    Función para implementar el algoritmo de árbol de decisión.
    """
    is_imbalanced = check_imbalance()
    # Comprueba desbalanceo.

    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    if is_imbalanced:  # Si hay desbalanceo...
        x_train, y_train = over_under_sampling(x_train, y_train)
        # Balancea train.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev)
    # Preprocesa.

    if args.debug:  # Si debug activo...
        try:
            train_debug = x_train.copy()
            # Copia x_train.

            train_debug[args.prediction] = y_train.values
            # Añade etiquetas.

            dev_debug = x_dev.copy()
            # Copia x_dev.

            dev_debug[args.prediction] = y_dev.values
            # Añade etiquetas.

            train_debug.to_csv('output/train-processed.csv', index=False)
            # Guarda train procesado.

            dev_debug.to_csv('output/dev-processed.csv', index=False)
            # Guarda dev procesado.

            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
            # Mensaje de éxito.

        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            # Mensaje de error.

            print(e)
            # Error concreto.

    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        # Crea barra de progreso.

        gs = GridSearchCV(
            RandomForestClassifier(),
            # Aquí hay un error conceptual:
            # debería ser DecisionTreeClassifier() si realmente quiere entrenar un árbol de decisión.

            args.decision_tree,
            # Parámetros del árbol sacados del JSON.

            cv=5,
            # 5 folds.

            n_jobs=args.cpu,
            # CPUs.

            scoring=get_scoring_metrics(),
            # Métricas.

            refit="score"
            # Reentrena con la métrica principal.
        )

        start_time = time.time()
        # Tiempo inicial.

        gs.fit(x_train, y_train)
        # Entrena.

        end_time = time.time()
        # Tiempo final.

        for i in range(100):
            # Bucle artificial para animar la barra.

            time.sleep(random.uniform(0.06, 0.15))
            # Pausa pequeña aleatoria.

            pbar.update(random.random()*2)
            # Actualiza barra.

        pbar.n = 100
        # Fuerza 100%.

        pbar.last_print_n = 100
        # Ajusta impresión.

        pbar.update(0)
        # Refresca.

    execution_time = end_time - start_time
    # Calcula tiempo total.

    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    # Lo muestra.

    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra resultados.

    save_model(gs,"decision_tree")
    # Guarda el modelo.

def random_forest():  # Función para entrenar Random Forest.
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    """
    is_imbalanced = check_imbalance()
    # Comprueba desbalanceo.

    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide datos.

    if is_imbalanced:
        x_train, y_train = over_under_sampling(x_train, y_train)
        # Balancea train si hace falta.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev)
    # Preprocesa.

    if args.debug:
        try:
            train_debug = x_train.copy()
            # Copia x_train.

            train_debug[args.prediction] = y_train.values
            # Añade y_train.

            dev_debug = x_dev.copy()
            # Copia x_dev.

            dev_debug[args.prediction] = y_dev.values
            # Añade y_dev.

            train_debug.to_csv('output/train-processed.csv', index=False)
            # Guarda train.

            dev_debug.to_csv('output/dev-processed.csv', index=False)
            # Guarda dev.

            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
            # Informa.

        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            # Mensaje de error.

            print(e)
            # Error concreto.

    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        # Barra de progreso.

        gs = GridSearchCV(
            RandomForestClassifier(),
            # Modelo Random Forest.

            args.random_forest,
            # Hiperparámetros del JSON.

            cv=5,
            # 5 folds.

            n_jobs=args.cpu,
            # CPUs.

            scoring=get_scoring_metrics(),
            # Métricas.

            refit="score"
            # Métrica principal para refit.
        )

        start_time = time.time()
        # Tiempo inicial.

        gs.fit(x_train, y_train)
        # Entrena.

        end_time = time.time()
        # Tiempo final.

        for i in range(100):
            # Bucle artificial de la barra.

            time.sleep(random.uniform(0.06, 0.15))
            # Pausa.

            pbar.update(random.random()*2)
            # Actualiza barra.

        pbar.n = 100
        # Fuerza 100%.

        pbar.last_print_n = 100
        # Ajusta impresión.

        pbar.update(0)
        # Refresca.

    execution_time = end_time - start_time
    # Calcula tiempo.

    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    # Lo muestra.

    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra métricas.

    save_model(gs, "random_forest")
    # Guarda modelo.

def naive_bayes():  # Función para entrenar Naive Bayes.
    """
    Función para implementar el algoritmo Naive Bayes.
    """
    is_imbalanced = check_imbalance()
    # Comprueba desbalanceo.

    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide datos.

    if is_imbalanced:
        x_train, y_train = over_under_sampling(x_train, y_train)
        # Balancea train.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev)
    # Preprocesa.

    if args.debug:
        try:
            train_debug = x_train.copy()
            # Copia x_train.

            train_debug[args.prediction] = y_train.values
            # Añade etiquetas.

            dev_debug = x_dev.copy()
            # Copia x_dev.

            dev_debug[args.prediction] = y_dev.values
            # Añade etiquetas.

            train_debug.to_csv('output/train-processed.csv', index=False)
            # Guarda train procesado.

            dev_debug.to_csv('output/dev-processed.csv', index=False)
            # Guarda dev procesado.

            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
            # Mensaje de éxito.

        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            # Mensaje de error.

            print(e)
            # Error concreto.

    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:
        # Crea barra de progreso.

        gs = GridSearchCV(
            GaussianNB(),
            # Modelo Naive Bayes gaussiano.

            args.naive_bayes,
            # Hiperparámetros del JSON.

            cv=5,
            # 5 folds.

            n_jobs=args.cpu,
            # CPUs.

            scoring=get_scoring_metrics(),
            # Métricas.

            refit="score"
            # Reentrena según score principal.
        )

        start_time = time.time()
        # Guarda tiempo inicial.

        gs.fit(x_train, y_train)
        # Entrena el modelo.

        end_time = time.time()
        # Guarda tiempo final.

        for i in range(100):
            # Bucle artificial de barra.

            time.sleep(random.uniform(0.06, 0.15))
            # Pausa pequeña aleatoria.

            pbar.update(random.random() * 2)
            # Actualiza barra.

        pbar.n = 100
        # Fuerza progreso a 100.

        pbar.last_print_n = 100
        # Ajusta última impresión.

        pbar.update(0)
        # Refresca.

    execution_time = end_time - start_time
    # Calcula tiempo total.

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    # Muestra tiempo.

    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra métricas.

    save_model(gs, "naive_bayes")
    # Guarda modelo.

# ======================= PROGRAMA PRINCIPAL =======================  # Separador del main.

if __name__ == "__main__":  # Esto asegura que este bloque solo se ejecute si el archivo se lanza directamente.
    np.random.seed(42)
    # Fija la semilla aleatoria de numpy para reproducibilidad.

    print("=== Clasificador ===")
    # Imprime el título del programa.

    signal.signal(signal.SIGINT, signal_handler)
    # Asocia la señal Ctrl+C a la función signal_handler.

    args = parse_args()
    # Lee argumentos de terminal y JSON.

    print("\n- Creando carpeta output...")
    # Muestra mensaje.

    try:
        os.makedirs('output')
        # Intenta crear la carpeta output.

        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
        # Mensaje si se crea bien.

    except FileExistsError:
        # Si la carpeta ya existe...

        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
        # Lo informa.

    except Exception as e:
        # Si ocurre otro error...

        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        # Muestra error.

        print(e)
        # Error concreto.

        sys.exit(1)
        # Sale del programa.

    print("\n- Cargando datos...")
    # Mensaje informativo.

    data = load_data(args.file)
    # Carga el CSV de datos y lo guarda en la variable global data.

    print("\n- Descargando diccionarios...")
    # Mensaje informativo.

    nltk.download('stopwords')
    # Descarga el recurso stopwords.

    nltk.download('punkt')
    # Descarga el recurso necesario para tokenizar.

    nltk.download('wordnet')
    # Descarga wordnet, aunque en este código realmente no se usa después.

    print("\n- Ejecutando algoritmo...")
    # Mensaje informativo.

    if args.algorithm == "kNN":
        # Si en el JSON o argumentos se ha pedido kNN...

        try:
            kNN()
            # Ejecuta el entrenamiento de kNN.

            print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
            # Mensaje de éxito.

            sys.exit(0)
            # Sale correctamente.

        except Exception as e:
            print(e)
            # Si falla, imprime el error.

    elif args.algorithm == "decision_tree":
        # Si se ha pedido árbol de decisión...

        try:
            decision_tree()
            # Ejecuta esa función.

            print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
            # Mensaje de éxito.

            sys.exit(0)
            # Sale correctamente.

        except Exception as e:
            print(e)
            # Imprime error.

    elif args.algorithm == "random_forest":
        # Si se ha pedido random forest...

        try:
            random_forest()
            # Ejecuta esa función.

            print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
            # Mensaje de éxito.

            sys.exit(0)
            # Sale correctamente.

        except Exception as e:
            print(e)
            # Imprime error.

    elif args.algorithm == "naive_bayes":
        # Si se ha pedido naive bayes...

        try:
            naive_bayes()
            # Ejecuta esa función.

            print(Fore.GREEN+"Algoritmo naive bayes ejecutado con éxito"+Fore.RESET)
            # Mensaje de éxito.

            sys.exit(0)
            # Sale correctamente.

        except Exception as e:
            print(e)
            # Imprime error.

    else:
        # Si el algoritmo indicado no coincide con ninguno de los soportados...

        print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
        # Muestra mensaje de error.

        sys.exit(1)
        # Sale con error.
