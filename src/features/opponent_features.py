import pandas as pd


def add_opponent_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add opponent state and relative-difference features
    for the two-player game.
    """

    required_columns = [
        "episode_id",
        "player",
        "money",
        "plant_tile_count",
        "pasture_tile_count",
        "weed_tile_count",
        "total_plant_yield",
        "total_pasture_yield",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = df.copy()

    result = result.sort_values(
        ["episode_id", "player"]
    ).copy()

    # Opponent state
    result["opponent_money"] = (
        result.groupby("episode_id")["money"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    result["opponent_plant_count"] = (
        result.groupby("episode_id")["plant_tile_count"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    result["opponent_pasture_count"] = (
        result.groupby("episode_id")["pasture_tile_count"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    result["opponent_weed_count"] = (
        result.groupby("episode_id")["weed_tile_count"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    result["opponent_plant_yield"] = (
        result.groupby("episode_id")["total_plant_yield"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    result["opponent_pasture_yield"] = (
        result.groupby("episode_id")["total_pasture_yield"]
        .transform(
            lambda x: x.iloc[::-1].to_numpy()
        )
    )

    # Relative state
    result["money_difference"] = (
        result["money"]
        - result["opponent_money"]
    )

    result["plant_difference"] = (
        result["plant_tile_count"]
        - result["opponent_plant_count"]
    )

    result["pasture_difference"] = (
        result["pasture_tile_count"]
        - result["opponent_pasture_count"]
    )

    result["weed_difference"] = (
        result["weed_tile_count"]
        - result["opponent_weed_count"]
    )

    result["plant_yield_difference"] = (
        result["total_plant_yield"]
        - result["opponent_plant_yield"]
    )

    result["pasture_yield_difference"] = (
        result["total_pasture_yield"]
        - result["opponent_pasture_yield"]
    )

    return result