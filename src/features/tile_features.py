def extract_tile_features(tiles: list) -> dict:
    """
    Convert a farm tile grid into aggregate numerical features.
    """

    features = {
        "locked_tile_count": 0,
        "empty_tile_count": 0,
        "plant_tile_count": 0,
        "pasture_tile_count": 0,
        "weed_tile_count": 0,

        "carrot_plant_count": 0,
        "melon_plant_count": 0,
        "strawberry_plant_count": 0,
        "tomato_plant_count": 0,
        "wheat_plant_count": 0,

        "cow_count": 0,
        "sheep_count": 0,
        "goose_count": 0,

        "total_plant_yield": 0,
        "total_pasture_yield": 0,

        "watered_plant_count": 0,
        "unwatered_plant_count": 0,
        "fertilized_plant_count": 0,

        "fed_animal_count": 0,
        "unfed_animal_count": 0,
        "cared_animal_count": 0,
        "uncared_animal_count": 0,
    }

    for tile_row in tiles:
        for tile in tile_row:

            # Empty tile
            if tile is None:
                features["empty_tile_count"] += 1
                continue

            # Locked tile
            if tile == "LOCKED":
                features["locked_tile_count"] += 1
                continue

            # Ignore unexpected values safely
            if not isinstance(tile, dict):
                continue

            kind = tile.get("kind")

            # -------------------------
            # WEED
            # -------------------------
            if kind == "WEED":
                features["weed_tile_count"] += 1

            # -------------------------
            # PLANT
            # -------------------------
            elif kind == "PLANT":

                features["plant_tile_count"] += 1

                crop = tile.get("crop")

                crop_key_map = {
                    "CARROT": "carrot_plant_count",
                    "MELON": "melon_plant_count",
                    "STRAWBERRY": "strawberry_plant_count",
                    "TOMATO": "tomato_plant_count",
                    "WHEAT": "wheat_plant_count",
                }

                if crop in crop_key_map:
                    features[crop_key_map[crop]] += 1

                features["total_plant_yield"] += (
                    tile.get("yield_units", 0)
                )

                if tile.get("watered_today", False):
                    features["watered_plant_count"] += 1
                else:
                    features["unwatered_plant_count"] += 1

                if tile.get("fertilized_until_day") is not None:
                    features["fertilized_plant_count"] += 1

            # -------------------------
            # PASTURE
            # -------------------------
            elif kind == "PASTURE":

                features["pasture_tile_count"] += 1

                animal = tile.get("animal")

                animal_key_map = {
                    "COW": "cow_count",
                    "SHEEP": "sheep_count",
                    "GOOSE": "goose_count",
                }

                if animal in animal_key_map:
                    features[animal_key_map[animal]] += 1

                features["total_pasture_yield"] += (
                    tile.get("yield_units", 0)
                )

                if tile.get("fed_today", False):
                    features["fed_animal_count"] += 1
                else:
                    features["unfed_animal_count"] += 1

                if tile.get("cared_today", False):
                    features["cared_animal_count"] += 1
                else:
                    features["uncared_animal_count"] += 1

    return features