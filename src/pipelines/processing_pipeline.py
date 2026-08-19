from pathlib import Path
import gc

import pandas as pd

from src.data.ingestion import (
    get_json_files,
    load_json,
)
from src.data.validation import validate_episode
from src.features.extractor import extract_episode
from src.utils.logger import get_logger


logger = get_logger(__name__)


def process_all_episodes(
    input_folder: str | Path,
    output_folder: str | Path,
    batch_size: int = 25,
) -> None:
    """
    Process JSON episodes in batches and save
    each batch as a Parquet file.
    """

    json_files = get_json_files(input_folder)

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Found %s JSON files",
        len(json_files),
    )

    batch_rows = []
    part_number = 1
    successful_files = 0
    failed_files = 0

    for index, file_path in enumerate(
        json_files,
        start=1,
    ):

        try:
            logger.info(
                "[%s/%s] Processing %s",
                index,
                len(json_files),
                file_path.name,
            )

            # 1. Load
            data = load_json(file_path)

            # 2. Validate
            validate_episode(data)

            # 3. Extract
            rows = extract_episode(data)

            # 4. Add records to batch
            batch_rows.extend(rows)

            successful_files += 1

            # Release JSON/episode references
            del data
            del rows

            # 5. Write batch
            if successful_files % batch_size == 0:

                _write_batch(
                    batch_rows=batch_rows,
                    output_folder=output_folder,
                    part_number=part_number,
                )

                batch_rows.clear()

                gc.collect()

                part_number += 1

        except Exception as error:

            failed_files += 1

            logger.exception(
                "FAILED: %s | Error: %s",
                file_path.name,
                error,
            )

    # Write remaining records
    if batch_rows:

        _write_batch(
            batch_rows=batch_rows,
            output_folder=output_folder,
            part_number=part_number,
        )

        batch_rows.clear()
        gc.collect()

    logger.info("Processing complete")
    logger.info(
        "Successful files: %s",
        successful_files,
    )
    logger.info(
        "Failed files: %s",
        failed_files,
    )


def _write_batch(
    batch_rows: list[dict],
    output_folder: Path,
    part_number: int,
) -> None:
    """
    Convert one batch of records to a DataFrame
    and save it as Parquet.
    """

    logger.info(
        "Creating DataFrame with %s rows",
        len(batch_rows),
    )

    batch_df = pd.DataFrame(batch_rows)

    output_path = (
        output_folder
        / f"part-{part_number:04d}.parquet"
    )

    logger.info(
        "Writing %s",
        output_path,
    )

    batch_df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Saved batch shape: %s",
        batch_df.shape,
    )

    del batch_df