"""Table input for mup3d keyword blocks.

Each keyword takes a ``components`` table (one row per component per block)
and an optional ``settings`` table (one row per block), both as pandas
DataFrames, plus keyword arguments for options shared by all blocks. Column
names follow the PHREEQC 3 manual. The ``block`` column holds the block
number written after the keyword (``SOLUTION 10``), and ``note`` is the
description written after it. An empty cell (NaN) means the value is not
given, so the field is left out of the script and PHREEQC uses its default.
"""

import warnings

import numpy as np
import pandas as pd

DICT_DEPRECATION = (
    "{name}: dict input is deprecated and will be removed in v1.0.0; "
    "pass a components DataFrame instead"
)

SOLUTION_COMPONENT_COLUMNS = {
    "required": ["block", "element", "conc"],
    "optional": ["units", "as", "gfw", "redox", "charge", "phase", "phase_si"],
}
# allowed columns of the settings table; pH and pe are rows of the components table
SOLUTION_SETTING_COLUMNS = [
    "block", "note", "temp", "pressure", "redox", "units", "density", "density_calculate",
]
# keyword options shared by all solutions; a solution's own value overrides them
SOLUTION_OPTIONS = [
    "temp", "pressure", "pH", "pe", "redox", "units", "density", "density_calculate",
]
EQUILIBRIUM_PHASE_COMPONENT_COLUMNS = {
    "required": ["block", "phase"],
    "optional": ["si", "m0", "alternative", "constraint", "force_equality"],
}
EQUILIBRIUM_PHASE_SETTING_COLUMNS = ["block", "note"]

# columns whose values must be numbers
SOLUTION_NUMERIC = ("conc", "gfw", "phase_si", "temp", "pressure", "density", "pH", "pe")
# columns whose values must be True or False
SOLUTION_FLAGS = ("density_calculate",)
EQUILIBRIUM_PHASE_NUMERIC = ("si", "m0")

# PHREEQC defaults for EQUILIBRIUM_PHASES lines
DEFAULT_SI = 0.0
DEFAULT_M0 = 10.0
CONSTRAINTS = ("dissolve_only", "precipitate_only")

# pH and pe are component rows, so that they can carry charge or phase
_SOLUTION_SPECIAL = ("pH", "pe")


def warn_dict_input(name, stacklevel=3):
    """Warn that dict input to a keyword block is deprecated."""
    warnings.warn(DICT_DEPRECATION.format(name=name), DeprecationWarning,
                  stacklevel=stacklevel)


def is_given(value):
    """Return True unless ``value`` is None or NaN."""
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def as_flag(value, column):
    """Read a boolean cell; empty means False."""
    if not is_given(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"Column '{column}' must be True or False, got {value!r}")


def as_number(value, what):
    """Read a numeric cell, naming the cell in the error."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None


def as_block_number(value):
    """Read a block number; it must be a non-negative integer."""
    if isinstance(value, (bool, np.bool_)) or not is_given(value):
        raise ValueError(f"Block numbers must be non-negative integers, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Block numbers must be non-negative integers, got {value!r}") from None
    if not number.is_integer() or number < 0:
        raise ValueError(f"Block numbers must be non-negative integers, got {value!r}")
    return int(number)


def check_columns(df, required, optional, table):
    """Check that ``df`` is a DataFrame with known and required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{table} must be a pandas DataFrame, got {type(df).__name__}")
    allowed = list(required) + list(optional)
    unknown = [col for col in df.columns if col not in allowed]
    if unknown:
        raise ValueError(f"Unknown column(s) {unknown} in {table}; allowed columns are {allowed}")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {table}")


def row_values(row, columns):
    """Return the given (non-NaN) cells of a row, keyed by column."""
    return {col: row[col] for col in columns if col in row.index and is_given(row[col])}


def read_settings(settings, columns, numeric, table, flags=()):
    """Read a settings table into ``{block: {column: value}}``.

    Only given cells are kept. ``note`` is stripped to a single line.
    """
    values = {}
    if settings is None:
        return values
    optional = [col for col in columns if col != "block"]
    check_columns(settings, ["block"], optional, table)
    for _, row in settings.iterrows():
        number = as_block_number(row["block"])
        if number in values:
            raise ValueError(f"Block {number} appears more than once in {table}")
        given = row_values(row, optional)
        for key in given:
            if key in numeric:
                given[key] = as_number(given[key], f"'{key}' for block {number} in {table}")
            elif key in flags:
                given[key] = as_flag(given[key], key)
        if "note" in given:
            given["note"] = " ".join(str(given["note"]).split())
        values[number] = given
    return values


def check_options(options, allowed, numeric, keyword, flags=()):
    """Check keyword-level options against the allowed settings."""
    unknown = [key for key in options if key not in allowed]
    if unknown:
        raise TypeError(f"{keyword} got unknown option(s) {unknown}; allowed options are {list(allowed)}")
    checked = {}
    for key, value in options.items():
        if key in numeric:
            value = as_number(value, f"{keyword} option '{key}'")
        elif key in flags:
            value = as_flag(value, key)
        checked[key] = value
    return checked


def check_density(block_numbers, options, block_settings):
    """Check that ``density_calculate`` comes with a ``density`` value."""
    for number in block_numbers:
        merged = {**options, **block_settings.get(number, {})}
        if merged.get("density_calculate") and "density" not in merged:
            raise ValueError(f"Solution {number}: 'density_calculate' needs a 'density' value")


def fmt(value):
    """Format a number the same way as the dict-input block writers."""
    return f"{float(value):.5e}"


def header(keyword, number, note=None):
    """Return the first line of a keyword block."""
    line = f"{keyword} {number}"
    if note:
        line += f" {note}"
    return line + "\n"


def warn_settings_only(block_settings, components, label, keyword, written_with):
    """Warn about blocks that are in the settings table but not in the components table.

    Such a block is still written, because it may be intended (PHREEQC accepts
    an empty block), but it is more often a typo in a block number.
    """
    only = sorted(set(block_settings) - set(components))
    if only:
        warnings.warn(
            f"{label}(s) {only} are in the settings table but not in the components "
            f"table; {keyword} {', '.join(map(str, only))} will be written with {written_with}",
            UserWarning, stacklevel=4,
        )


def read_solution_tables(components, settings=None):
    """Read the Solutions components and settings tables.

    Returns
    -------
    block_numbers : list of int
        Solution numbers, sorted.
    entries : dict
        ``{number: {element: {column: value}}}`` in table order.
    block_settings : dict
        ``{number: {setting: value}}``, including ``note``.
    """
    table = "Solutions components"
    check_columns(components, SOLUTION_COMPONENT_COLUMNS["required"],
                  SOLUTION_COMPONENT_COLUMNS["optional"], table)
    block_settings = read_settings(settings, SOLUTION_SETTING_COLUMNS, SOLUTION_NUMERIC,
                                   "Solutions settings", SOLUTION_FLAGS)

    entries = {}
    for _, row in components.iterrows():
        number = as_block_number(row["block"])
        element = row["element"]
        if not is_given(element):
            raise ValueError(f"Empty 'element' in {table} for block {number}")
        element = str(element).strip()
        if not is_given(row["conc"]):
            raise ValueError(f"Empty 'conc' for {element} in solution {number}")
        block = entries.setdefault(number, {})
        if element in block:
            raise ValueError(f"{element} appears more than once in solution {number}")

        # block and element identify the row; the other columns describe it
        entry = row_values(row, ["conc"] + SOLUTION_COMPONENT_COLUMNS["optional"])
        for key in entry:
            if key in SOLUTION_NUMERIC:
                entry[key] = as_number(entry[key], f"'{key}' for {element} in solution {number}")
        entry["charge"] = as_flag(row.get("charge"), "charge")
        if "as" in entry and "gfw" in entry:
            raise ValueError(f"{element} in solution {number}: give either 'as' or 'gfw', not both")
        if entry["charge"] and "phase" in entry:
            raise ValueError(f"{element} in solution {number}: give either 'charge' or 'phase', not both")
        if "phase_si" in entry and "phase" not in entry:
            raise ValueError(f"{element} in solution {number}: 'phase_si' needs 'phase'")
        if element in _SOLUTION_SPECIAL:
            extra = [col for col in ("units", "as", "gfw", "redox") if col in entry]
            if extra:
                raise ValueError(f"{element} in solution {number} cannot have {extra}")
        block[element] = entry

    for number, block in entries.items():
        charged = [element for element, entry in block.items() if entry["charge"]]
        if len(charged) > 1:
            raise ValueError(f"Solution {number} has more than one charge balance: {charged}")

    block_numbers = sorted(set(entries) | set(block_settings))
    if not block_numbers:
        raise ValueError("Solutions tables define no solutions")
    warn_settings_only(block_settings, entries, "Solution", "SOLUTION",
                       "settings only (pure water, PHREEQC defaults)")
    return block_numbers, entries, block_settings


def apply_charge_flag(number, entries, settings, species):
    """Put the ``add_charge_flag`` charge balance on one row of a solution.

    The flag goes through the same checks as the table's ``charge`` column:
    one charge balance per solution, and not on a row with ``phase``. A
    ``pH`` or ``pe`` given as a keyword option becomes a row, so that it can
    carry the flag.

    Parameters
    ----------
    number : int
        Solution number, for error messages.
    entries : dict
        ``{element: {column: value}}`` for this solution. Not modified.
    settings : dict
        Merged settings for this solution. Not modified.
    species : str
        The row that gets the charge balance, e.g. ``"pH"`` or ``"Cl"``.

    Returns
    -------
    entries, settings : dict
        Copies with the flag applied.
    applied : bool
        False if the solution has no ``species``, so nothing was changed.

    Raises
    ------
    ValueError
        If the solution would have more than one charge balance, or the
        ``species`` row has ``phase``.
    """
    entries = {element: dict(entry) for element, entry in entries.items()}
    settings = dict(settings)
    if species not in entries and species in _SOLUTION_SPECIAL and species in settings:
        entries = {species: {"conc": settings.pop(species), "charge": False}, **entries}
    if species not in entries:
        return entries, settings, False
    if "phase" in entries[species]:
        raise ValueError(f"{species} in solution {number}: add_charge_flag cannot go on a row "
                         "with 'phase'; give either 'charge' or 'phase', not both")
    entries[species]["charge"] = True
    charged = [element for element, entry in entries.items() if entry["charge"]]
    if len(charged) > 1:
        raise ValueError(f"Solution {number} has more than one charge balance: {charged} "
                         f"(add_charge_flag={species!r} and the 'charge' column)")
    return entries, settings, True


def solution_block(number, entries, settings):
    """Write a SOLUTION block from table input.

    Parameters
    ----------
    number : int
        Solution number.
    entries : dict
        ``{element: {column: value}}`` for this solution, in table order.
    settings : dict
        Settings for this solution, already merged with keyword options and
        defaults (``units``, ``temp``, ...). ``note`` is written after the
        solution number as the PHREEQC description.

    Returns
    -------
    str
        The SOLUTION block.
    """
    script = header("SOLUTION", number, settings.get("note"))
    for key in ("units", "temp", "pressure", "density", "redox"):
        if key in settings:
            value = settings[key]
            if key == "temp":
                value = str(float(value))
            elif key in ("pressure", "density"):
                value = fmt(value)
            if key == "density" and settings.get("density_calculate"):
                value += " calculate"
            script += f"    {key} {value}\n"
    for key in _SOLUTION_SPECIAL:
        if key in settings:
            script += f"    {key} {fmt(settings[key])}\n"
    for element, entry in entries.items():
        line = f"    {element} {fmt(entry['conc'])}"
        if "units" in entry:
            line += f" {entry['units']}"
        if "as" in entry:
            line += f" as {entry['as']}"
        if "gfw" in entry:
            line += f" gfw {fmt(entry['gfw'])}"
        if "redox" in entry:
            line += f" {entry['redox']}"
        if entry["charge"]:
            line += " charge"
        if "phase" in entry:
            line += f" {entry['phase']}"
            if "phase_si" in entry:
                line += f" {fmt(entry['phase_si'])}"
        script += line + "\n"
    script += "END\n"
    return script


def read_equilibrium_phase_tables(components, settings=None):
    """Read the EquilibriumPhases components and settings tables.

    Returns
    -------
    block_numbers : list of int
        EQUILIBRIUM_PHASES numbers, sorted.
    data : dict
        ``{position: {phase: {"si": .., "m0": ..}}}`` with positions
        ``0..n-1`` in the order of ``block_numbers`` (the dict-input
        structure).
    extras : dict
        ``{number: {phase: {column: value}}}`` with the given optional
        columns other than ``si`` and ``m0``.
    block_settings : dict
        ``{number: {setting: value}}``, i.e. ``note``.
    """
    table = "EquilibriumPhases components"
    check_columns(components, EQUILIBRIUM_PHASE_COMPONENT_COLUMNS["required"],
                  EQUILIBRIUM_PHASE_COMPONENT_COLUMNS["optional"], table)
    block_settings = read_settings(settings, EQUILIBRIUM_PHASE_SETTING_COLUMNS, (),
                                   "EquilibriumPhases settings")

    blocks, extras, defaulted = {}, {}, []
    for _, row in components.iterrows():
        number = as_block_number(row["block"])
        phase = row["phase"]
        if not is_given(phase):
            raise ValueError(f"Empty 'phase' in {table} for block {number}")
        phase = str(phase).strip()
        block = blocks.setdefault(number, {})
        if phase in block:
            raise ValueError(f"{phase} appears more than once in EQUILIBRIUM_PHASES {number}")

        given = row_values(row, EQUILIBRIUM_PHASE_COMPONENT_COLUMNS["optional"])
        for key in EQUILIBRIUM_PHASE_NUMERIC:
            if key in given:
                given[key] = as_number(given[key], f"'{key}' for {phase} in EQUILIBRIUM_PHASES {number}")
        for key, default in (("si", DEFAULT_SI), ("m0", DEFAULT_M0)):
            if key not in given:
                defaulted.append(f"{key} = {default} for {phase} in EQUILIBRIUM_PHASES {number}")
        block[phase] = {
            "si": given.pop("si", DEFAULT_SI),
            "m0": given.pop("m0", DEFAULT_M0),
        }
        if "constraint" in given and given["constraint"] not in CONSTRAINTS:
            raise ValueError(f"{phase} in EQUILIBRIUM_PHASES {number}: 'constraint' must be one of "
                             f"{list(CONSTRAINTS)}, got {given['constraint']!r}")
        if "force_equality" in given:
            given["force_equality"] = as_flag(given["force_equality"], "force_equality")
        if given:
            extras.setdefault(number, {})[phase] = given

    if defaulted:
        warnings.warn(
            "EquilibriumPhases: empty cells use the PHREEQC defaults: " + "; ".join(defaulted),
            UserWarning, stacklevel=3,
        )
    block_numbers = sorted(set(blocks) | set(block_settings))
    if not block_numbers:
        raise ValueError("EquilibriumPhases tables define no blocks")
    warn_settings_only(block_settings, blocks, "EquilibriumPhases block", "EQUILIBRIUM_PHASES",
                       "every phase at m0 = 0 (fill_missing_minerals)")
    data = {pos: blocks.get(number, {}) for pos, number in enumerate(block_numbers)}
    return block_numbers, data, extras, block_settings


def equilibrium_phases_block(number, phases, extras=None, note=None):
    """Write an EQUILIBRIUM_PHASES block from table input.

    Parameters
    ----------
    number : int
        Block number.
    phases : dict
        ``{phase: {"si": .., "m0": ..}}``.
    extras : dict, optional
        ``{phase: {column: value}}`` with ``alternative``, ``constraint`` and
        ``force_equality``. Phases without extras are written as
        ``phase si m0``.
    note : str, optional
        Description written after the block number.

    Returns
    -------
    str
        The EQUILIBRIUM_PHASES block.
    """
    extras = extras or {}
    script = header("EQUILIBRIUM_PHASES", number, note)
    for phase, values in phases.items():
        extra = extras.get(phase, {})
        line = f"    {phase} {fmt(values['si'])}"
        if "alternative" in extra:
            line += f" {extra['alternative']}"
        line += f" {fmt(values['m0'])}"
        if "constraint" in extra:
            line += f" {extra['constraint']}"
        script += line + "\n"
        if "force_equality" in extra:
            script += f"        -force_equality {str(extra['force_equality']).lower()}\n"
    script += "END\n"
    return script
