# Features used by the ML model.
# Keep this as the single source of truth for
# both training and prediction.

MODEL_FEATURES = [
    "player",
    "step",
    "day",
    "hour",
    "money",
    "farmer_x",
    "farmer_y",
    "hires_today",
    "remaining_overage_time",

    "hands_action_count",
    "market_action_count",

    # Market
    "carrot_price",
    "egg_price",
    "fertilizer_price",
    "melon_price",
    "milk_price",
    "strawberry_price",
    "tomato_price",
    "wheat_price",
    "wool_price",

    # Seeds
    "carrot_seed",
    "melon_seed",
    "strawberry_seed",
    "tomato_seed",
    "wheat_seed",

    # Shed
    "shed_carrot",
    "shed_cow",
    "shed_egg",
    "shed_fertilizer",
    "shed_goose",
    "shed_melon",
    "shed_milk",
    "shed_sheep",
    "shed_strawberry",
    "shed_tomato",
    "shed_wheat",
    "shed_wool",

    # Farm / tiles
    "locked_tile_count",
    "empty_tile_count",
    "plant_tile_count",
    "pasture_tile_count",
    "weed_tile_count",

    "carrot_plant_count",
    "melon_plant_count",
    "strawberry_plant_count",
    "tomato_plant_count",
    "wheat_plant_count",

    "cow_count",
    "sheep_count",
    "goose_count",

    "total_plant_yield",
    "total_pasture_yield",

    "watered_plant_count",
    "unwatered_plant_count",
    "fertilized_plant_count",

    "fed_animal_count",
    "unfed_animal_count",
    "cared_animal_count",
    "uncared_animal_count",

    # Opponent state
    "opponent_money",
    "opponent_plant_count",
    "opponent_pasture_count",
    "opponent_weed_count",
    "opponent_plant_yield",
    "opponent_pasture_yield",

    # Relative state
    "money_difference",
    "plant_difference",
    "pasture_difference",
    "weed_difference",
    "plant_yield_difference",
    "pasture_yield_difference",
]


TARGET_COLUMN = "final_reward"
GROUP_COLUMN = "episode_id"