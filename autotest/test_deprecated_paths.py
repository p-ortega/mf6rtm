import pickle

import pytest

from mf6rtm import config, utils
from mf6rtm.io import yaml_reader
from mf6rtm.mup3d import SurfacePhases


@pytest.mark.parametrize(
    "old_module, name, new_module",
    [
        ("mf6rtm.config.config", "MF6RTMConfig", config),
        ("mf6rtm.config.yaml_reader", "load_yaml_to_phreeqcrm", yaml_reader),
        ("mf6rtm.utils.utils", "get_indices", utils),
    ],
)
def test_old_module_path_warns_and_resolves(old_module, name, new_module):
    with pytest.warns(DeprecationWarning, match=old_module):
        obj = getattr(__import__(old_module, fromlist=[name]), name)
    assert obj is getattr(new_module, name)


def test_old_from_package_import_still_works():
    with pytest.warns(DeprecationWarning, match="mf6rtm.utils.utils"):
        from mf6rtm.utils import utils as old_utils
        old_utils.get_indices
    with pytest.warns(DeprecationWarning, match="mf6rtm.config.config"):
        from mf6rtm.config.config import MF6RTMConfig
    assert MF6RTMConfig is config.MF6RTMConfig


def test_surfaces_alias_warns():
    from mf6rtm import mup3d

    with pytest.warns(DeprecationWarning, match="SurfacePhases"):
        assert mup3d.Surfaces is SurfacePhases
    with pytest.warns(DeprecationWarning, match="SurfacePhases"):
        assert mup3d.base.Surfaces is SurfacePhases


def test_old_pickle_loads():
    # Protocol 0 writes class paths as plain text, so the new names can be
    # swapped for the ones an old mup3d.pkl holds
    data = pickle.dumps(
        {"surfaces_phases": SurfacePhases({0: {"Hfo": [0.1, 600]}}),
         "config": config.MF6RTMConfig()},
        protocol=0,
    )
    data = data.replace(b"mf6rtm.mup3d.base\nSurfacePhases\n", b"mf6rtm.mup3d.base\nSurfaces\n")
    data = data.replace(b"mf6rtm.config\nMF6RTMConfig\n", b"mf6rtm.config.config\nMF6RTMConfig\n")
    assert b"\nSurfaces\n" in data and b"mf6rtm.config.config\n" in data

    with pytest.warns(DeprecationWarning):
        loaded = pickle.loads(data)
    assert type(loaded["surfaces_phases"]) is SurfacePhases
    assert type(loaded["config"]) is config.MF6RTMConfig
