from configs.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
)

from src.pipelines.processing_pipeline import (
    process_all_episodes,
)


if __name__ == "__main__":

    process_all_episodes(
        input_folder=RAW_DATA_DIR,
        output_folder=PROCESSED_DATA_DIR,
        batch_size=BATCH_SIZE,
    )