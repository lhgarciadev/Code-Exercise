import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_dataset():
    """
    Carga el dataset y lo preprocesa para su uso en un modelo de clasificación.
    El dataset se carga desde un archivo JSONL y se transforma en un DataFrame de pandas.
    Se añaden nuevas características al DataFrame, como la frecuencia del vendedor, el modo de envío,
    la cantidad de fotos, la longitud del título, etc.
    Luego, se separa el conjunto de datos en características (X) y etiquetas (y), y se codifican las etiquetas.
    Finalmente, se divide el conjunto de datos en conjuntos de entrenamiento y prueba.
    El conjunto de entrenamiento se utiliza para entrenar el modelo y el conjunto de prueba se utiliza
    para evaluar su rendimiento.
    Returns:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (np.ndarray): Conjunto de etiquetas de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (np.ndarray): Conjunto de etiquetas de prueba.
        label_encoder (LabelEncoder): Codificador de etiquetas para transformar etiquetas de texto en números.
        seller_ids_test (pd.Series): IDs de los vendedores en el conjunto de prueba.
        y_true_labels (pd.Series): Etiquetas verdaderas del conjunto de prueba.
    Raises:
        FileNotFoundError: Si el archivo de datos no se encuentra en la ruta especificada.
        ValueError: Si el formato del archivo de datos no es válido o si hay errores en los datos.
    """
    print("Cargando dataset...")

    # Cargar el dataset desde un archivo JSONL
    data_path = os.getenv("DATA_PATH", "data/MLA_100k_checked_v3.jsonlines")

    # Comprobar si el archivo existe
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # convertir a DataFrame
    df = pd.DataFrame(data)

    # Preprocesar el DataFrame

    # Calcular la frecuencia relativa de cada vendedor en el dataset
    df["seller_freq"] = df["seller_id"].map(
        df["seller_id"].value_counts(normalize=True)
    )

    # Identificar si el envío es gratuito (1) o no (0)
    df["is_free_shipping"] = (
        df["shipping"]
        .apply(
            lambda x: x.get("free_shipping", False) if isinstance(x, dict) else False
        )
        .astype(int)
    )

    # Extraer el modo de envío (por ejemplo, "me2", "custom", etc.)
    df["shipping_mode"] = df["shipping"].apply(
        lambda x: x.get("mode", "unknown") if isinstance(x, dict) else "unknown"
    )

    # Identificar si el producto permite recogida local (1) o no (0)
    df["local_pick_up"] = (
        df["shipping"]
        .apply(
            lambda x: x.get("local_pick_up", False) if isinstance(x, dict) else False
        )
        .astype(int)
    )

    # Agrupar la cantidad de métodos de pago no asociados a MercadoPago (0-3 o "4+")
    df["payment_method_group"] = (
        df["non_mercado_pago_payment_methods"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .apply(lambda n: n if n <= 3 else "4+")
    )

    # Indicar si el ítem tiene tags relacionados con visitas arrastradas (1) o no (0)
    df["has_dragged_visits"] = (
        df["tags"]
        .apply(
            lambda x: (
                "dragged_visits" in x or "dragged_bids_and_visits" in x
                if isinstance(x, list)
                else False
            )
        )
        .astype(int)
    )

    # Verificar si el ítem tiene una miniatura de buena calidad
    df["has_good_thumbnail"] = (
        df["tags"]
        .apply(
            lambda x: "good_quality_thumbnail" in x if isinstance(x, list) else False
        )
        .astype(int)
    )

    # Verificar si el ítem tiene una miniatura de mala calidad
    df["has_poor_thumbnail"] = (
        df["tags"]
        .apply(
            lambda x: "poor_quality_thumbnail" in x if isinstance(x, list) else False
        )
        .astype(int)
    )

    # Indicar si el ítem fue republicado automáticamente (relist)
    df["was_relisted"] = (
        df["tags"]
        .apply(lambda x: "free_relist" in x if isinstance(x, list) else False)
        .astype(int)
    )

    # Agrupar ítems según la cantidad de imágenes: 0, 1, 2-6, o 7+
    df["picture_group"] = (
        df["pictures"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .apply(
            lambda x: (
                "0" if x == 0 else ("1" if x == 1 else ("2-6" if x <= 6 else "7+"))
            )
        )
    )

    # Clasificar el largo del título como corto, medio o largo
    df["title_length_group"] = (
        df["title"]
        .str.len()
        .apply(lambda x: "short" if x < 30 else ("medium" if x < 60 else "long"))
    )

    # Indicar si el título contiene la palabra "nuevo"
    df["title_contains_new"] = (
        df["title"].str.lower().str.contains("nuevo").fillna(False).astype(int)
    )

    # Indicar si el título contiene la palabra "usado"
    df["title_contains_used"] = (
        df["title"].str.lower().str.contains("usado").fillna(False).astype(int)
    )

    # Extraer el ID del estado del vendedor
    df["state_id"] = df["seller_address"].apply(
        lambda x: x["state"]["id"] if isinstance(x, dict) else "unknown"
    )

    # Calcular la diferencia entre el precio actual y el precio base
    df["price_diff"] = df["price"] - df["base_price"]

    # Calcular el ratio de ventas sobre el total de unidades ofrecidas
    df["sold_ratio"] = df["sold_quantity"] / (
        df["sold_quantity"] + df["available_quantity"] + 1e-5
    )

    # Calcular el cambio en la disponibilidad del inventario
    df["availability_change"] = df["initial_quantity"] - df["available_quantity"]

    # Calcular el porcentaje del inventario inicial que se ha utilizado
    df["stock_used_ratio"] = df["availability_change"] / (df["initial_quantity"] + 1e-5)

    # Indicar si la fecha de última actualización difiere de la de creación (1 = actualizado)
    df["was_updated"] = (
        pd.to_datetime(df["last_updated"], errors="coerce")
        != pd.to_datetime(df["date_created"], errors="coerce")
    ).astype(int)

    # Indicar si el producto sigue activo
    df["is_active"] = df["status"].apply(lambda x: 1 if x == "active" else 0)

    # Seleccionar columnas relevantes
    cols_base = [
        "currency_id",
        "base_price",
        "price",
        "listing_type_id",
        "initial_quantity",
        "sold_quantity",
        "available_quantity",
        "condition",
        "seller_freq",
        "is_free_shipping",
        "shipping_mode",
        "local_pick_up",
        "payment_method_group",
        "has_dragged_visits",
        "has_good_thumbnail",
        "has_poor_thumbnail",
        "was_relisted",
        "picture_group",
        "title_length_group",
        "title_contains_new",
        "title_contains_used",
        "state_id",
        "price_diff",
        "sold_ratio",
        "availability_change",
        "stock_used_ratio",
        "was_updated",
        "is_active",
    ]

    # Filtrar columnas
    df_model = df[cols_base].copy()
    seller_ids = df["seller_id"]
    y = df_model["condition"]
    X = df_model.drop(columns=["condition"])

    # Convertir a categorías
    X_encoded = pd.get_dummies(X, drop_first=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["used", "new"])
    y_encoded = label_encoder.transform(y)

    # Dividir el conjunto de datos en entrenamiento y prueba
    N = -10000
    X_train = X_encoded[:N]
    X_test = X_encoded[N:]
    y_train = y_encoded[:N]
    y_test = y_encoded[N:]
    seller_ids_test = seller_ids[N:]
    y_true_labels = y[N:]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder,
        seller_ids_test.reset_index(drop=True),
        y_true_labels.reset_index(drop=True),
    )


if __name__ == "__main__":
    build_dataset()
