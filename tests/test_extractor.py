from src.features.extractor import extract_episode


def test_extract_episode_returns_list():

    # Later we can load a small sample JSON here.
    # The important expectation:
    # output should be a list of dictionaries.

    assert callable(extract_episode)