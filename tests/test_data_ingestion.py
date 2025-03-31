import json
import pytest
import pandas as pd
from itemclassifier.stage_01_data_ingestion import build_dataset


@pytest.fixture
def sample_file(tmp_path):
    """Crea un archivo temporal con una muestra válida del dataset."""
    sample_data = {
        "seller_address": {"state": {"name": "Capital Federal", "id": "AR-C"}},
        "shipping": {
            "local_pick_up": True,
            "free_shipping": False,
            "mode": "not_specified",
        },
        "non_mercado_pago_payment_methods": [{"id": "MLATB"}, {"id": "MLAWC"}],
        "seller_id": 12345,
        "tags": ["dragged_bids_and_visits"],
        "pictures": [{"id": "img1"}, {"id": "img2"}],
        "title": "Producto nuevo barato",
        "price": 100.0,
        "base_price": 90.0,
        "initial_quantity": 10,
        "sold_quantity": 2,
        "available_quantity": 8,
        "last_updated": "2023-01-01T12:00:00Z",
        "date_created": "2023-01-01T10:00:00Z",
        "status": "active",
        "currency_id": "ARS",
        "listing_type_id": "bronze",
        "condition": "new",
    }

    file_path = tmp_path / "sample.jsonlines"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data) + "\n")

    return file_path


def test_build_dataset_from_sample(
    sample_file, monkeypatch
):  # pylint: disable=redefined-outer-name
    """Valida que build_dataset funcione con un archivo mínimo."""
    monkeypatch.setenv("DATA_PATH", str(sample_file))

    X_train, _, X_test, y_test, label_encoder, seller_ids_test, _ = build_dataset()

    assert isinstance(X_train, pd.DataFrame)
    assert X_train.empty  # Solo una muestra => cae en X_test
    assert X_test.shape[0] == 1
    assert y_test[0] in [0, 1]
    assert label_encoder.inverse_transform([y_test[0]])[0] == "new"
    assert seller_ids_test.iloc[0] == 12345
