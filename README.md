
# Ejercicio de Código - itemclassifier

[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Estado](https://img.shields.io/badge/estado-en%20progreso-yellow)](https://shields.io/)

## Descripción

Este proyecto es parte de un ejercicio para el rol de **Data Scientist** en el contexto del Marketplace de MercadoLibre. El objetivo principal es construir un modelo de Machine Learning que prediga si un ítem publicado en el Marketplace es **nuevo o usado**, basándose en las características disponibles.

El dataset principal se encuentra en `data/MLA_100k_checked_v3.jsonlines` y cuenta con 100,000 registros de ítems verificados.

## Funcionalidades

* Ingesta de datos desde archivos `.jsonlines`
* EDA (Exploratory Data Analysis)
* Procesamiento y limpieza de datos
* Entrenamiento de modelos de clasificación
* Evaluación con métricas como **Accuracy** y **F1-Score**
* Serialización de modelos y artefactos con `joblib`
* Notebooks con análisis paso a paso

## Comenzando

Estas instrucciones te permitirán obtener una copia del proyecto y ejecutarlo localmente para desarrollo o pruebas.

### Estructura del Proyecto

```
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── MLA_100k_checked_v3.jsonlines
│   └── predicciones_con_seller_id.csv
├── models
│   └── tree-based
│       ├── label_encoder.pkl
│       └── xgboost.pkl
├── notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preparation.ipynb
│   └── 04_model_training.ipynb
├── poetry.lock
├── pyproject.toml
├── pytest.ini
├── src
│   └── itemclassifier/
│       ├── __main__.py
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_model_training.py
│       └── stage_03_model_evaluation.py
└── tests/
    ├── test_data_ingestion.py
    ├── test_model_evaluation.py
    └── test_model_training.py
```

> El proyecto incluye un `Makefile` para facilitar tareas comunes como limpieza de artefactos, ejecución de pruebas y formateo de código. Usa `make help` para ver los comandos disponibles.

### Requisitos Previos

* Python 3.10
* Poetry para la gestión de dependencias
* Entorno compatible con Jupyter Notebooks

### Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/lhgarciadev/Code-Exercise.git
    ```

2. Navega al directorio del proyecto:
    ```bash
    cd Code-Exercise
    ```

3. Instala las dependencias usando Poetry:
    ```bash
    poetry install
    ```

4. Activa el entorno:
    ```bash
    poetry shell
    ```

## Ejecución del Código

Puedes ejecutar el pipeline completo utilizando el punto de entrada principal:

```bash
poetry run itemclassifier
```

También puedes ejecutar notebooks de forma individual dentro de `notebooks/`:

* `01_data_ingestion.ipynb`
* `02_eda.ipynb`
* `03_data_preparation.ipynb`
* `04_model_training.ipynb`

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
poetry run pytest tests/ --cov=src
```

## Mejoras Futuras

* Implementación de una API REST para servir el modelo
* Automatizar el pipeline completo con un orquestador como Airflow

## Autor

* **Leonardo H. Garcia Diaz** - [@lhgarciadev](https://github.com/lhgarciadev)
