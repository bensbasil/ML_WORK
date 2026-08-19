from src.features.tile_features import extract_tile_features
from src.features.opponent_features import add_opponent_features
import pandas as pd

from src.features.tile_features import extract_tile_features
from src.features.opponent_features import add_opponent_features


def extract_episode(data: dict) -> list[dict]:
    """
    Convert one nested episode into flat feature records.

    Each record represents:
    one player at one timestep.
    """

    rows = []

    episode_id = data["id"]
    final_rewards = data["rewards"]

    for step_number, step_data in enumerate(data["steps"]):

        for player, agent in enumerate(step_data):

            obs = agent["observation"]

            # Current player's farm
            farm = obs["farms"][player]

            # -------------------------
            # FARM
            # -------------------------
            farmer_x, farmer_y = farm["farmer"]

            # -------------------------
            # PRIVATE STATE
            # -------------------------
            private = obs["private"]
            seeds = private["seeds"]
            shed = private["shed"]

            # -------------------------
            # MARKET
            # -------------------------
            prices = obs["market"]["prices"]

            # -------------------------
            # TILES
            # -------------------------
            tile_features = extract_tile_features(
                farm["tiles"]
            )

            # -------------------------
            # BASE FEATURES
            # -------------------------
            row = {
                "episode_id": episode_id,

                "player": player,
                "step": step_number,
                "day": obs["day"],
                "hour": obs["hour"],

                "money": farm["money"],
                "farmer_x": farmer_x,
                "farmer_y": farmer_y,
                "hires_today": farm["hires_today"],

                "remaining_overage_time": obs.get(
                    "remainingOverageTime"
                ),

                "step_reward": agent["reward"],
                "status": agent["status"],
                "final_reward": final_rewards[player],

                "farmer_action": ",".join(map(str, agent["action"]["farmer"])),
                "hands_action_count": len(
                    agent["action"]["hands"]
                ),
                "market_action_count": len(
                    agent["action"]["market"]
                ),

                # Market
                "carrot_price": prices["CARROT"],
                "egg_price": prices["EGG"],
                "fertilizer_price": prices["FERTILIZER"],
                "melon_price": prices["MELON"],
                "milk_price": prices["MILK"],
                "strawberry_price": prices["STRAWBERRY"],
                "tomato_price": prices["TOMATO"],
                "wheat_price": prices["WHEAT"],
                "wool_price": prices["WOOL"],

                # Seeds
                "carrot_seed": seeds["CARROT"],
                "melon_seed": seeds["MELON"],
                "strawberry_seed": seeds["STRAWBERRY"],
                "tomato_seed": seeds["TOMATO"],
                "wheat_seed": seeds["WHEAT"],

                # Shed
                "shed_carrot": shed["CARROT"],
                "shed_cow": shed["COW"],
                "shed_egg": shed["EGG"],
                "shed_fertilizer": shed["FERTILIZER"],
                "shed_goose": shed["GOOSE"],
                "shed_melon": shed["MELON"],
                "shed_milk": shed["MILK"],
                "shed_sheep": shed["SHEEP"],
                "shed_strawberry": shed["STRAWBERRY"],
                "shed_tomato": shed["TOMATO"],
                "shed_wheat": shed["WHEAT"],
                "shed_wool": shed["WOOL"],
            }

            # Add tile features
        
            row.update(tile_features)

            rows.append(row)

    # Convert base rows to DataFrame
    episode_df = pd.DataFrame(rows)

    # Add opponent-relative features
    episode_df = add_opponent_features(
        episode_df
    )

    return episode_df.to_dict(
        orient="records"
    )