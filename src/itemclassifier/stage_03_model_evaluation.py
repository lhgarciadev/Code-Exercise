import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from itemclassifier.stage_01_data_ingestion import build_dataset


def evaluate_model():
    """
    Función para evaluar el modelo de clasificación de artículos.
    Carga el conjunto de datos de prueba y el modelo entrenado, realiza
    predicciones y calcula métricas de rendimiento como la precisión y el F1 score.
    Guarda las predicciones y el label encoder en archivos CSV y pickle, respectivamente.
    También imprime las métricas de rendimiento en la consola.
    Returns:
        results_file (Path): Ruta del archivo CSV que contiene las predicciones.
        label_encoder_file (Path): Ruta del archivo pickle que contiene el label encoder.
    Raises:
        FileNotFoundError: Si el archivo de datos o el modelo no se encuentran en las rutas configurables.
        ValueError: Si el formato del archivo de datos no es válido o si hay errores en los datos.
    """
    print("Cargando datos y modelo para evaluación...")
    # Cargar el conjunto de datos
    _, _, X_test, y_test, label_encoder, seller_ids_test, y_true_labels = (
        build_dataset()
    )

    # Rutas configurables
    model_path = Path(os.getenv("MODEL_PATH", "models/tree-based"))
    data_path = Path(os.getenv("DATA_PATH", "data"))

    # Cargar el modelo
    model_file = model_path / "xgboost.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_file}")

    model = joblib.load(model_file)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de rendimiento
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Convertir etiquetas predichas a texto
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Asegurar que el directorio existe
    data_path.mkdir(parents=True, exist_ok=True)
    results_file = data_path / "predicciones_con_seller_id.csv"

    # Guardar resultados en un DataFrame
    results_df = pd.DataFrame(
        {
            "seller_id": seller_ids_test,
            "actual_condition": y_true_labels,
            "predicted_condition": y_pred_labels,
        }
    )

    results_df.to_csv(results_file, index=False)
    print(f"Predicciones exportadas a: {results_file}")

    # Guardar label encoder
    model_path.mkdir(parents=True, exist_ok=True)
    label_encoder_file = model_path / "label_encoder.pkl"
    joblib.dump(label_encoder, label_encoder_file)
    print(f"Label encoder guardado en: {label_encoder_file}")

    return results_file, label_encoder_file


if __name__ == "__main__":
    evaluate_model()
