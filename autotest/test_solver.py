import numpy as np
from unittest.mock import MagicMock

from mf6rtm.simulation.solver import (
    Mf6RTM,
    get_conc_change_mask,
    get_inactive_idx,
    get_less_than_zero_idx,
    longest_common_substring,
)


def _make_mock_mf6rtm(reactive=True, timing='all', tsteps=None, kper=1, kstp=1):
    """Minimal mock Mf6RTM for is_reactive_tstep (no model run)."""
    mock = MagicMock()
    mock.reactive = reactive
    mock.mf6api.kper = kper
    mock.mf6api.kstp = kstp
    mock.config.reactive = {'timing': timing, 'tsteps': tsteps or []}
    return mock


class TestIsReactiveTstep:
    """Test suite for Mf6RTM.is_reactive_tstep timing logic."""

    def test_non_reactive_returns_false(self):
        mock = _make_mock_mf6rtm(reactive=False)
        assert Mf6RTM.is_reactive_tstep(mock) is False

    def test_timing_all_returns_true(self):
        mock = _make_mock_mf6rtm(timing='all')
        assert Mf6RTM.is_reactive_tstep(mock) is True

    def test_timing_user_matching_tstep(self):
        # current_tstep is built as [kper, kstp], so tsteps must hold a matching list
        mock = _make_mock_mf6rtm(timing='user', tsteps=[[1, 2]], kper=1, kstp=2)
        assert Mf6RTM.is_reactive_tstep(mock) is True

    def test_timing_user_non_matching_tstep(self):
        mock = _make_mock_mf6rtm(timing='user', tsteps=[[9, 9]], kper=1, kstp=2)
        assert Mf6RTM.is_reactive_tstep(mock) is False

    def test_unknown_timing_warns_and_defaults_true(self, capsys):
        mock = _make_mock_mf6rtm(timing='adaptive')
        result = Mf6RTM.is_reactive_tstep(mock)
        assert result is True
        assert 'Unknown strategy' in capsys.readouterr().out


class TestReplaceInactiveCells:
    """Test suite for _replace_inactive_cells (cells skipped by PhreeqcRM)."""

    def test_inactive_cells_replaced_with_current_conc(self):
        """Check that inactive (skipped) cells are assigned the concentrations 
        computed at the end of the current transport time step, 
        rather than those from the previous time step.
        """
        mock = MagicMock()
        mock.phreeqcbmi.ncomps = 2
        mock.previous_iteration_conc = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        mock.current_iteration_conc = np.array([[1.5, 1.6, 1.7], [2.5, 2.6, 2.7]])
        from_phreeqc = np.full((2, 3), 9.0)
        diffmask = np.array([1, 0, 0])  # cell 0 active, cells 1 and 2 skipped
        result = Mf6RTM._replace_inactive_cells(mock, from_phreeqc, diffmask)
        np.testing.assert_allclose(result[:, 0], [9.0, 9.0])
        np.testing.assert_allclose(result[:, 1:], mock.current_iteration_conc[:, 1:])


class TestGetConcChangeMask:
    """Test suite for get_conc_change_mask (active/inactive cell mask)."""

    def test_shape_and_binary_values(self):
        """Mask is 1D of length nxyz with values only in {0, 1}."""
        ci = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        ck = np.array([[1.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
        mask = get_conc_change_mask(ck, ci)
        assert mask.shape == (3,)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_no_change_gives_all_zero(self):
        """Identical current/previous arrays -> no cell flagged active."""
        ci = np.array([[1.0, 5.0, 3.0], [2.0, 2.0, 2.0]])
        ck = ci.copy()
        mask = get_conc_change_mask(ck, ci)
        np.testing.assert_array_equal(mask, np.zeros(3))

    def test_single_component_change_flags_cell(self):
        """A change in any one component activates that cell only."""
        ci = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        ck = np.array([[1.0, 2.0, 1.0], [2.0, 2.0, 2.0]])  # cell 1, comp 0 changed
        mask = get_conc_change_mask(ck, ci)
        np.testing.assert_array_equal(mask, np.array([0, 1, 0]))

    def test_division_by_zero_is_safe(self):
        """Zeros in the previous array must not produce NaN/inf."""
        ci = np.array([[0.0, 1.0, 0.0]])
        ck = np.array([[5.0, 1.0, 0.0]])
        mask = get_conc_change_mask(ck, ci)
        assert np.isfinite(mask).all()
        assert set(np.unique(mask)).issubset({0, 1})

    def test_threshold_sensitivity(self):
        """A small relative change is active under a tight threshold and
        inactive under a loose one."""
        ci = np.array([[1.0]])
        ck = np.array([[1.0 + 1e-6]])
        # default tight threshold -> change is significant -> active
        assert get_conc_change_mask(ck, ci)[0] == 1
        # loose threshold swallows the change -> inactive
        assert get_conc_change_mask(ck, ci, threshold=1e-3)[0] == 0


class TestGetInactiveIdx:
    """Test suite for get_inactive_idx (cells >= sentinel value)."""

    def test_returns_indices_at_or_above_val(self):
        arr = np.array([1.0, 1e30, 5.0, 2e30])
        assert get_inactive_idx(arr) == [1, 3]

    def test_returns_list_type(self):
        arr = np.array([1e30, 0.0])
        assert isinstance(get_inactive_idx(arr), list)

    def test_empty_when_none_inactive(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert get_inactive_idx(arr) == []

    def test_custom_threshold(self):
        arr = np.array([1.0, 10.0, 100.0])
        assert get_inactive_idx(arr, val=10.0) == [1, 2]


class TestGetLessThanZeroIdx:
    """Test suite for get_less_than_zero_idx (negative-value locations)."""

    def test_returns_tuple(self):
        arr = np.array([-1.0, 2.0, -3.0])
        idx = get_less_than_zero_idx(arr)
        assert isinstance(idx, tuple)

    def test_one_dimensional_indices(self):
        arr = np.array([-1.0, 2.0, -3.0, 4.0])
        np.testing.assert_array_equal(get_less_than_zero_idx(arr)[0], [0, 2])

    def test_two_dimensional_indices(self):
        arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
        rows, cols = get_less_than_zero_idx(arr)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [1, 0])

    def test_empty_when_no_negatives(self):
        arr = np.array([0.0, 1.0, 2.0])
        assert len(get_less_than_zero_idx(arr)[0]) == 0


class TestLongestCommonSubstring:
    """Test suite for longest_common_substring (GWT model-name stem)."""

    def test_common_stem(self):
        assert longest_common_substring(["gwtca", "gwtcl", "gwtna"]) == "gwt"

    def test_longer_shared_prefix(self):
        assert longest_common_substring(["GWT_ca", "GWT_cl"]) == "GWT_c"

    def test_empty_list_returns_empty(self):
        assert longest_common_substring([]) == ""

    def test_no_common_substring(self):
        assert longest_common_substring(["abc", "xyz"]) == ""

    def test_case_sensitive(self):
        assert longest_common_substring(["GWT", "gwt"]) == ""
