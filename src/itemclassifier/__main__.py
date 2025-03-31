import argparse


def main():
    parser = argparse.ArgumentParser(description="ItemClassifier CLI")
    parser.add_argument(
        "stage",
        choices=["ingestion", "train", "evaluation"],
        help="Stage to run: ingestion, train or evaluation",
    )
    args = parser.parse_args()

    if args.stage == "ingestion":
        from itemclassifier.stage_01_data_ingestion import build_dataset

        print("Ejecutando etapa de ingestión de datos...")
        _ = build_dataset()
        print("Ingestión de datos completada exitosamente.")

    elif args.stage == "train":
        from itemclassifier.stage_02_model_training import train_and_save_model

        train_and_save_model()

    elif args.stage == "evaluation":
        from itemclassifier.stage_03_model_evaluation import evaluate_model

        evaluate_model()


if __name__ == "__main__":
    main()
