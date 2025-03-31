import joblib
from pathlib import Path
from xgboost import XGBClassifier
from itemclassifier.stage_01_data_ingestion import build_dataset


def simple_train_model(model_path: Path):
    """Entrena un modelo XGB simple y lo guarda en model_path."""
    X_train, y_train, *_ = build_dataset()
    model = XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    model_path.mkdir(parents=True, exist_ok=True)
    file = model_path / "xgboost.pkl"
    joblib.dump(model, file)
    return file


def test_model_training_creates_pkl(tmp_path, monkeypatch):
    """Valida que el modelo se entrene y guarde correctamente."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))

    model_file = simple_train_model(Path(tmp_path))

    assert model_file.exists()
    model = joblib.load(model_file)
    assert isinstance(model, XGBClassifier)
