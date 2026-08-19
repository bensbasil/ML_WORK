from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = 76
EXPECTED_ROWS_PER_EPISODE = 1440


def get_parquet_files(folder_path: str | Path) -> list[Path]:
    """Return all Parquet files in the processed-data folder."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Processed folder not found: {folder}"
        )

    return sorted(folder.glob("*.parquet"))


def validate_file(
    file_path: Path,
    expected_columns: int = EXPECTED_COLUMNS,
    expected_rows: int = EXPECTED_ROWS_PER_EPISODE,
) -> dict:
    """Validate one processed Parquet file."""

    df = pd.read_parquet(file_path)

    result = {
        "file": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "valid": True,
        "errors": [],
    }

    if len(df.columns) != expected_columns:
        result["valid"] = False
        result["errors"].append(
            f"Expected {expected_columns} columns, "
            f"found {len(df.columns)}"
        )

    if len(df) != expected_rows:
        result["valid"] = False
        result["errors"].append(
            f"Expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    if result["missing_values"] > 0:
        result["valid"] = False
        result["errors"].append(
            f"Found {result['missing_values']} missing values"
        )

    return result


def validate_dataset(processed_dir: str | Path) -> pd.DataFrame:
    """Validate every processed Parquet file."""

    parquet_files = get_parquet_files(processed_dir)

    print(f"Found {len(parquet_files)} Parquet files")

    results = []

    for index, file_path in enumerate(parquet_files, start=1):

        print(
            f"Validating {index}/{len(parquet_files)}: "
            f"{file_path.name}"
        )

        try:
            result = validate_file(file_path)
            results.append(result)

        except Exception as error:
            results.append({
                "file": file_path.name,
                "rows": None,
                "columns": None,
                "missing_values": None,
                "duplicate_rows": None,
                "valid": False,
                "errors": [str(error)],
            })

    return pd.DataFrame(results)