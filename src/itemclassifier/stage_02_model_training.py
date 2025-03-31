import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from itemclassifier.stage_01_data_ingestion import build_dataset


def train_and_save_model():
    """
    Carga el dataset y entrena un modelo de XGBoost utilizando GridSearchCV.
    El modelo se entrena utilizando validación cruzada y se guardan los mejores
    hiperparámetros encontrados.
    El modelo entrenado se guarda en un archivo utilizando joblib.
    El archivo se guarda en una ruta configurable a través de la variable de entorno MODEL_PATH.
    Se utiliza un clasificador XGBoost con los siguientes hiperparámetros:
    - n_estimators: número de árboles a construir
    - max_depth: profundidad máxima de los árboles
    - learning_rate: tasa de aprendizaje
    - subsample: proporción de muestras a utilizar para entrenar cada árbol
    Se imprimen los mejores hiperparámetros encontrados y se guarda el modelo entrenado
    en un archivo llamado "xgboost.pkl".
    Returns:
        model_file (Path): Ruta del archivo donde se guarda el modelo entrenado.
    Raises:
        FileNotFoundError: Si el archivo de datos no se encuentra en la ruta especificada.
        ValueError: Si el formato del archivo de datos no es válido o si hay errores en los datos.
    """
    print("Cargando dataset para entrenamiento...")
    # Cargar el dataset
    X_train, y_train, _, _, _, _, _ = build_dataset()

    # Separar las características y etiquetas
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 12],
        "learning_rate": [0.1, 0.3, 0.6],
        "subsample": [0.4, 0.8, 1.0],
    }

    # Definir el modelo XGBoost
    grid_xgb = GridSearchCV(
        estimator=XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid=param_grid_xgb,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    print("Entrenando XGBoost con GridSearchCV...")
    # Entrenar el modelo utilizando GridSearchCV
    grid_xgb.fit(X_train, y_train)
    best_model = grid_xgb.best_estimator_
    print("Mejores hiperparámetros encontrados:", grid_xgb.best_params_)

    # Ruta configurable para guardar el modelo
    model_path = Path(os.getenv("MODEL_PATH", "models/tree-based"))
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "xgboost.pkl"

    # Guardar el modelo entrenado
    joblib.dump(best_model, model_file)
    print(f"Modelo guardado en '{model_file}'")

    return model_file


if __name__ == "__main__":
    train_and_save_model()
