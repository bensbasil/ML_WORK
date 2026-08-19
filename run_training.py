from pathlib import Path

from src.models.train import (
    load_step_data,
    prepare_data,
    split_by_episode,
    train_model,
    save_model,
)

from src.models.evaluate import (
    evaluate_model,
    save_metrics,
    save_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

STEP = 300


def main():

    print("=" * 50)
    print("STEP-300 CHAMPION MODEL")
    print("=" * 50)

    # -------------------------
    # 1. Load step-specific data
    # -------------------------
    print("\nLoading data...")

    df = load_step_data(
        processed_dir=PROCESSED_DIR,
        step=STEP,
    )

    print(f"Loaded rows: {len(df)}")

    # -------------------------
    # 2. Prepare X / y / groups
    # -------------------------
    X, y, groups = prepare_data(df)

    print(f"Features: {X.shape[1]}")

    # Keep identifiers for later error analysis
    metadata = df[
        [
            "episode_id",
            "player",
            "step",
        ]
    ].copy()

    # -------------------------
    # 3. Episode-level split
    # -------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = split_by_episode(
        X,
        y,
        groups,
    )

    # Metadata for exactly the same test rows
    test_metadata = metadata.iloc[test_idx].copy()

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # -------------------------
    # 4. Train
    # -------------------------
    print("\nTraining HistGradientBoosting...")

    model = train_model(
    model_name="hist_gradient_boosting",
    X_train=X_train,
    y_train=y_train,
    )

    # -------------------------
    # 5. Evaluate
    # -------------------------
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nEvaluation")
    print("-" * 30)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # -------------------------
    # 6. Save model
    # -------------------------
    model_path = (
    MODEL_DIR /
    f"champion_step_{STEP}.joblib"
    )

    save_model(
        model,
        model_path,
    )

    # -------------------------
    # 7. Save metrics
    # -------------------------
    metrics_path = (
        RESULTS_DIR /
        f"step_{STEP}_metrics.csv"
    )

    save_metrics(
        metrics,
        metrics_path,
    )

    # -------------------------
    # 8. Save predictions
    # -------------------------
    predictions = model.predict(X_test)

    predictions_path = (
        RESULTS_DIR /
        f"step_{STEP}_predictions.parquet"
    )

    save_predictions(
        y_test,
        predictions,
        test_metadata,
        predictions_path,
    )

    print("\nTraining pipeline complete ✅")


if __name__ == "__main__":
    main()