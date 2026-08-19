import pytest

from src.data.validation import validate_episode


def test_valid_episode():

    data = {
        "id": "episode_1",
        "rewards": [100, 200],
        "steps": [],
    }

    validate_episode(data)


def test_missing_keys():

    data = {
        "id": "episode_1",
    }

    with pytest.raises(ValueError):
        validate_episode(data)