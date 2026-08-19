from pathlib import Path

from src.models.train import (
    load_step_data,
    split_by_episode,
)

from src.models.train_gradient_boosting import (
    prepare_data,
    train_gradient_boosting,
    save_model,
)

from src.models.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

STEP = 300


def main():

    print("=" * 50)
    print("STEP-300 GRADIENT BOOSTING MODEL")
    print("=" * 50)

    print("\nLoading data...")

    df = load_step_data(
        processed_dir=PROCESSED_DIR,
        step=STEP,
    )

    X, y, groups = prepare_data(df)

    X_train, X_test, y_train, y_test = (
        split_by_episode(
            X,
            y,
            groups,
        )
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    print("\nTraining...")

    model = train_gradient_boosting(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nEvaluation")
    print("-" * 30)

    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    model_path = (
        MODEL_DIR /
        "gradient_boosting_step_300.joblib"
    )

    save_model(
        model,
        model_path,
    )

    print("\nTraining pipeline complete ✅")


if __name__ == "__main__":
    main()