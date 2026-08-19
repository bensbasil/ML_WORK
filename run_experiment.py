from pathlib import Path
import pandas as pd

from src.models.train import (
    load_step_data,
    prepare_data,
    split_by_episode,
    train_model,
    save_model,
)

from src.models.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
MODEL_DIR = PROJECT_ROOT / "models"

STEP = 300

MODELS = [
    "random_forest",
    "hist_gradient_boosting",
]


def main():

    print("=" * 60)
    print("STEP-300 MODEL COMPARISON")
    print("=" * 60)

    df = load_step_data(
        processed_dir=PROCESSED_DIR,
        step=STEP,
    )

    X, y, groups = prepare_data(df)

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

    print(f"Rows: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Train: {len(X_train)}")
    print(f"Test: {len(X_test)}")

    results = []

    for model_name in MODELS:

        print("\n" + "-" * 40)
        print(f"Training: {model_name}")
        print("-" * 40)

        model = train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(
            f"MAE : {metrics['MAE']:.4f}"
        )
        print(
            f"RMSE: {metrics['RMSE']:.4f}"
        )
        print(
            f"R²  : {metrics['R2']:.4f}"
        )

        results.append({
            "model": model_name,
            "step": STEP,
            **metrics,
        })

        model_path = (
            MODEL_DIR
            / f"{model_name}_step_{STEP}.joblib"
        )

        save_model(
            model,
            model_path,
        )

    results_df = (
    pd.DataFrame(results)
    .sort_values("R2", ascending=False)
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    # Save experiment results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_path = (
        results_dir / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nComparison saved to: {comparison_path}"
    )

if __name__ == "__main__":
    main()