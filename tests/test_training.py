from pathlib import Path

from footymind.features.etl_pipeline import run_etl
from footymind.models.train_model import run_training, TrainConfig
from footymind.utils.paths import get_model_path


def test_end_to_end_training_pipeline(tmp_path: Path):
    """
    Run ETL + training on the sample dataset and ensure a model artifact
    is produced without errors.
    """
    # Run ETL (idempotent; just overwrites processed CSVs)
    run_etl()

    # Train model (uses processed train.csv)
    cfg = TrainConfig(test_size=0.25)
    run_training(cfg)

    model_path = get_model_path()
    assert model_path.exists(), "Trained model artifact should exist after training."
