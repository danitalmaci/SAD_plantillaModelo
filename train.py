# -*- coding: utf-8 -*-
 
# ======================= PLANTILLA =======================


"""
Autores: Daniel Talmaci & June Castro
Script para la implementación de los siguientes algoritmos:
      1. kNN
      2. Decision Tree
      3. Random Forest
      4. Naïve Bayes
      
"""


# ======================= IMPORTS =======================

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore

# Sklearn
from sklearn.naive_bayes import GaussianNB
# from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Imblearn

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# ======================= PROGRAMA =======================


# ------------ Funciones auxiliares ------------

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
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree, random_forest o naive_bayes)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
   
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
   
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
   
    # Parseamos los argumentos
    return args
   
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para calcular métricas

# TODO Aqui poned lo que hayais hecho

def calculate_fscore(y_true, y_pred):
    fscore_micro = f1_score(y_true, y_pred, average='micro')
    fscore_macro = f1_score(y_true, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
       
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
       
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)



def get_missing_config():
    """
    Devuelve la configuración de tratamiento de valores perdidos.
    """
    return args.preprocessing.get("missing_values", {})

def get_scaling_config():
    """
    Devuelve la configuración de escalado.
    """
    return args.preprocessing.get("scaling", {})

def get_balancing_config():
    """
    Devuelve la configuración de balanceo.
    """
    return args.preprocessing.get("balancing", {})

def apply_missing_strategy(col, strategy_cfg, is_numeric=True):
    """
    Aplica la estrategia indicada en el JSON para los valores faltantes de una columna.
    """
    global data

    strategy = strategy_cfg.get("strategy", "none")

    if strategy == "drop_rows":
        data = data.dropna(subset=[col])
        print(f"Se eliminan filas con missing en '{col}'")
    elif strategy == "mean" and is_numeric:
        data[col] = data[col].fillna(data[col].mean())
        print(f"Se imputa la media en '{col}'")
    elif strategy == "median" and is_numeric:
        data[col] = data[col].fillna(data[col].median())
        print(f"Se imputa la mediana en '{col}'")
    elif strategy == "mode":
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
            print(f"Se imputa la moda en '{col}'")
    elif strategy == "constant":
        fill_value = strategy_cfg.get("value", 0 if is_numeric else "desconocido")
        data[col] = data[col].fillna(fill_value)
        print(f"Se imputa un valor constante en '{col}': {fill_value}")
    elif strategy == "none":
        print(f"No se aplica imputación en '{col}'")
    else:
        raise ValueError(f"Estrategia de missing no válida para '{col}': {strategy}")

def build_scaler(method):
    """
    Crea el scaler correspondiente al método indicado.
    """
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "maxabs":
        return MaxAbsScaler()
    if method in ["none", None]:
        return None
    raise ValueError(f"Método de escalado no soportado: {method}")


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes usando la configuración del JSON.
    """
    global data

    missing_cfg = get_missing_config()
    numeric_default = missing_cfg.get("numeric_default", {"strategy": "none"})
    categorical_default = missing_cfg.get("categorical_default", {"strategy": "none"})
    per_column = missing_cfg.get("per_column", {})

    for col in numerical_feature.columns:
        if data[col].isnull().sum() > 0:
            col_cfg = per_column.get(col, numeric_default)
            apply_missing_strategy(col, col_cfg, is_numeric=True)

    for col in categorical_feature.columns:
        if data[col].isnull().sum() > 0:
            col_cfg = per_column.get(col, categorical_default)
            apply_missing_strategy(col, col_cfg, is_numeric=False)

#TODO aqui preprocesar


def reescaler(numerical_feature):
    """
    Reescala las características numéricas usando la configuración del JSON.
    """
    global data

    scaling_cfg = get_scaling_config()
    default_method = scaling_cfg.get("default", "none")
    per_column = scaling_cfg.get("per_column", {})

    for col in numerical_feature.columns:
        method = per_column.get(col, default_method)
        scaler = build_scaler(method)

        if scaler is None:
            print(f"No se escala la columna {col}")
        else:
            data[col] = scaler.fit_transform(data[[col]])
            print(f"Columna {col} escalada con {method}")

#TODO aqui reescalar

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.
    """
    global data

    print("Conversión de variables categóricas a numéricas (Label Encoding)")

    for col in categorical_feature.columns:
        print(f"Codificando la columna {col}...")

        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
   
#TODO aqui lo que haga falta para pasar de categorial a numerico

def simplify_text(text_feature):
    """
    Simplifica el texto: minúsculas, quitar puntuación, tokenizar, eliminar stopwords y stemming.
    """
    global data

    print("Simplificando texto...")

    # stopwords en inglés (puedes cambiar a español si quieres)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for col in text_feature.columns:
        print(f"Procesando columna {col}...")

        # 1. Rellenar missing
        data[col] = data[col].fillna("")

        # 2. Pasar a minúsculas
        data[col] = data[col].str.lower()

        # 3. Quitar puntuación
        data[col] = data[col].str.translate(str.maketrans('', '', string.punctuation))

        # 4. Tokenizar + quitar stopwords + stemming
        def procesar_texto(texto):
            tokens = word_tokenize(texto)  # dividir en palabras
            tokens = [t for t in tokens if t not in stop_words]  # quitar stopwords
            tokens = [stemmer.stem(t) for t in tokens]  # stemming
            return " ".join(tokens)  # volver a string

        data[col] = data[col].apply(procesar_texto)
 #TODO aqui lo que sea preciso en caso de tener texto

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":              
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)


def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la configuración del JSON.
    """
    global data

    balancing_cfg = get_balancing_config()
    method = balancing_cfg.get("method", "none")

    if method == "none":
        print("No se aplica balanceo")
        return

    X = data.drop(columns=[args.prediction])
    y = data[args.prediction]
    random_state = balancing_cfg.get("random_state", 42)

    if method == "oversampling":
        sampler = RandomOverSampler(random_state=random_state)
        print("Aplicando Oversampling...")
    elif method == "undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
        print("Aplicando Undersampling...")
    else:
        raise ValueError(f"Método de balanceo no soportado: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    data = pd.concat([X_resampled, y_resampled], axis=1)

    print("Balanceo aplicado correctamente")

 
  #TODO aqui

def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing.get("drop_features", []), errors="ignore")
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        TODO 5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Simplificamos el texto
    simplify_text(text_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
   
    # Tratamos el texto
    process_text(text_feature)
   
    # Realizamos Oversampling o Undersampling
    over_under_sampling()

    drop_features()

    return data

# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    global data

    try:
        # Sacamos la columna a predecir
        X = data.drop(columns=[args.prediction])
        y = data[args.prediction]

        # Dividimos los datos
        x_train, x_dev, y_train, y_dev = train_test_split(
            X,
            y,
            test_size=args.split["test_size"],
            random_state=args.split["random_state"],
            stratify=y
        )

        print(Fore.GREEN + "Datos divididos con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Tamaño x_train:" + Fore.RESET, x_train.shape)
            print(Fore.MAGENTA + "> Tamaño x_dev:" + Fore.RESET, x_dev.shape)
            print(Fore.MAGENTA + "> Tamaño y_train:" + Fore.RESET, y_train.shape)
            print(Fore.MAGENTA + "> Tamaño y_dev:" + Fore.RESET, y_dev.shape)

        return x_train, x_dev, y_train, y_dev

    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)
 #TODO
 
 
def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
   
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            KNeighborsClassifier(),
            args.kNN,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
   
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
   
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
   
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        #TODO Llamar al decision trees
        #gs = GridSearchCV(
        gs = GridSearchCV(
            DecisionTreeClassifier(),
            args.decision_tree,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        # Barra de progreso (igual que en kNN)
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random()*2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
   
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
   
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
   
    # Guardamos el modelo utilizando pickle
    save_model(gs)
   
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
   
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
   
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        #TODO Llamar al decision trees
        #gs = GridSearchCV(
        gs = GridSearchCV(
            RandomForestClassifier(),
            args.random_forest,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        # Barra de progreso (igual que en kNN)
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random()*2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
   
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
   
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
def naive_bayes():
    """
    Función para implementar el algoritmo Naive Bayes.
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:
        
        gs = GridSearchCV(
            GaussianNB(),
            args.naive_bayes,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
       
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)
   
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
   
# ======================= PROGRAMA PRINCIPAL =======================

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":
            try:
                naive_bayes()
                print(Fore.GREEN+"Algoritmo naive bayes ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
