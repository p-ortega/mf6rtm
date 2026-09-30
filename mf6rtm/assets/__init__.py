from importlib import resources


def mrbeaker_path():
    with resources.path("mf6rtm.assets", "mrbeaker.png") as path:
        return path
