"""Configuration schema and (de)serialization for mf6rtm runs.

Defines :class:`MF6RTMConfig`, the container for reactive-transport run
settings, with helpers to load from and save to TOML and dictionary form.
"""
import toml
import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ConfigSchema:
    """Schema definition for configuration grouping."""
    group: str
    toml_name: Optional[str] = None  # Different name in TOML

class MF6RTMConfig:
    """MF6RTM Configuration class similar to FloPy package structure.
    This class provides a FloPy-style interface for configuring MF6RTM
    reaction timing parameters.

    Parameters
    ----------
    reaction_timing : str, optional
        Controls when reactions are calculated. Options:
        - 'all' : Calculate reactions at all time steps (default)
        - 'user' : Calculate reactions only at user-specified time steps
        - 'adaptive' : Use adaptive timing based on convergence criteria
    tsteps : List[Tuple[int, int]], optional
        List of (kper, kstp) tuples specifying when reactions should be calculated.
        Only used when reaction_timing='user'. Default is empty list.
        kper is stress period (1-based), kstp is time step (1-based).

    Attributes
    ----------
    reaction_timing : str
        Current reaction timing strategy.
    tsteps : List[Tuple[int, int]]
        List of time steps for reaction calculations.
    """
    # Config sections whose kwargs use the flat convention folded
    _SECTION_PREFIXES = ("reactive_", "emulator_", "output_", "solver_")

    def __init__(self, **kwargs):
        """Basic initialization."""
        # Minimal initialization
        for key, value in kwargs.items():
            self._ingest_kwarg(key, value)
        # Apply defaults for any missing attributes
        self._apply_defaults()

    def _ingest_kwarg(self, key, value):
        """Store a configuration kwarg.

        A ``<section>_<key>`` kwarg (e.g. ``reactive_externalio``) is folded into
        the nested section dict (``self.reactive['externalio']``) so the flat form
        accepted by ``set_config`` and the nested dicts the runtime reads never
        diverge.
        """
        for prefix in self._SECTION_PREFIXES:
            if key.startswith(prefix):
                section = prefix.rstrip("_")
                subkey = key[len(prefix):]
                sec = getattr(self, section, None)
                if not isinstance(sec, dict):
                    sec = {}
                    setattr(self, section, sec)
                sec[subkey] = value
                return
        setattr(self, key, value)

    def _validate_config(self):
        self._validate_reaction_timing()
        self._validate_tsteps()
        self.validated = True

    def _apply_defaults(self):
        """Apply default values for any missing attributes (using nested dicts)."""

        defaults = {
            "reactive": {
                "enabled": True,
                "timing": "all",
                "tsteps": [],
                "externalio": False,
            },
            "emulator": {
                "training_data": False,
                "feature_variables": [],
                "target_variables": [],
            },
            "output": {
                "output_format": "csv",
            },
            "solver": {
                "min_concentration": None,
                "no_react_cells": None,
            },
        }

        for section, section_defaults in defaults.items():
            # If the section (e.g. self.reactive) does not exist, create it
            if not hasattr(self, section) or getattr(self, section) is None:
                setattr(self, section, {})

            # Now fill in missing keys
            cfg_section = getattr(self, section)

            for key, default_value in section_defaults.items():
                cfg_section.setdefault(key, default_value)

    # def _apply_defaults(self):
    #     """Apply default values for any missing attributes."""
    #     defaults = {
    #         'reactive_enabled': True,
    #         'reactive_timing': 'all',
    #         'reactive_tsteps': [],
    #         'reactive_externalio': False,
    #         'emulator_training_data': False,
    #         'emulator_feature_variables': [],
    #         'emulator_target_variables': [],
    #     }

        # # Apply defaults for any missing attributes
        # for key, default_value in defaults.items():
        #     if not hasattr(self, key):
        #         setattr(self, key, default_value)

    def add_new_configuration(self, **kwargs):
        """Add new configuration parameters dynamically.

        Parameters
        ----------
        **kwargs
            Configuration parameters to set as attributes on the instance.
        """
        for key, value in kwargs.items():
            self._ingest_kwarg(key, value)
        # Update internal schema if needed
        self._update_schema_for_new_attrs(kwargs.keys())

    def _validate_reaction_timing(self):
        """Validate reaction_timing parameter."""
        valid_options = ['all', 'user', 'adaptive']
        if self.reactive['timing'] not in valid_options:
            raise ValueError(f"reaction_timing must be one of {valid_options}, "
                           f"got '{self.reactive['timing']}'")

    def _validate_tsteps(self):
        """Validate tsteps parameter."""
        if not isinstance(self.reactive['tsteps'], list):
            raise ValueError("tsteps must be a list")
        # error if self.reactive_tsteps is empty and timing is 'user'
        if self.reactive['timing'] == 'user' and len(self.reactive['tsteps']) == 0:
            raise ValueError("tsteps cannot be empty when reaction_timing is 'user'")
        if self.reactive['timing'] == 'all' and len(self.reactive['tsteps']) > 0:
            print("WARNING: Reactive time steps defined but timing set to 'all' instead of 'user'")
        normalized = []
        for i, tstep in enumerate(self.reactive['tsteps']):
            if not isinstance(tstep, (tuple, list)) or len(tstep) != 2:
                raise ValueError(f"tsteps[{i}] must be a tuple/list of length 2")

            kper, kstp = tstep
            if not isinstance(kper, int) or not isinstance(kstp, int):
                raise ValueError(f"tsteps[{i}] must contain integers")
            if kper < 1 or kstp < 1:
                raise ValueError(f"tsteps[{i}]: kper and kstp must be 1-indexed")
            normalized.append((kper, kstp))  # force into tuple
        # Ensure (1, 1) is included
        if (1, 1) not in normalized:
            normalized.insert(0, (1, 1))


    def get_tsteps_for_period(self, kper: int) -> List[int]:
        """Get time steps for a specific stress period.

        Parameters
        ----------
        kper : int
            Stress period number (1-based).

        Returns
        -------
        List[int]
            List of time step numbers for the given stress period.

        Examples
        --------
        >>> config = MF6RTMConfig(reactive_timing='user',
        ...                       reactive_tsteps=[(1, 1), (1, 10), (2, 5)])
        >>> config.get_tsteps_for_period(1)
        [1, 10]
        """
        return sorted([kstp for kp, kstp in self.reactive_tsteps if kp == kper])

    def is_reaction_tstep(self, kper: int, kstp: int) -> bool:
        """Check if reactions should be calculated at a specific time step.

        Parameters
        ----------
        kper : int
            Stress period number (1-based).
        kstp : int
            Time step number (1-based).

        Returns
        -------
        bool
            True if reactions should be calculated at this time step.

        Examples
        --------
        >>> config = MF6RTMConfig(reaction_timing='user', tsteps=[(1, 1)])
        >>> config.is_reaction_tstep(1, 1)
        True
        >>> config.is_reaction_tstep(1, 2)
        False
        """
        if self.reactive_timing == 'all':
            return True
        elif self.reactive_timing == 'user':
            return (kper, kstp) in self.reactive_tsteps
        elif self.reactive_timing == 'adaptive':
            # Placeholder for adaptive
            return True
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a nested dictionary for TOML output.

        Returns
        -------
        dict
            Nested mapping with category groups (``reactive``, ``emulator``),
            phase sections, and remaining keys, ordered for TOML serialization.
        """

        result = {}

        for attr_name, value in self.__dict__.items():
            if attr_name.startswith('_'):
                continue

            # Handle nested phase attributes
            if '_' in attr_name:
                parts = attr_name.split('_')
                if len(parts) >= 3 and parts[0] in ['equilibrium', 'kinetic', 'exchange'] and parts[-1] not in ['names']:
                    main_group = '_'.join(parts[:2])  # e.g., "equilibrium_phases"
                    sub_group = parts[2]
                    key = '_'.join(parts[3:]) if len(parts) > 3 else parts[2]
                    if main_group not in result:
                        result[main_group] = {}
                    if sub_group not in result[main_group]:
                        result[main_group][sub_group] = {}
                    result[main_group][sub_group][key] = value
                elif len(parts) >= 2 and parts[-1] in ['names']:
                    main_group = '_'.join(parts[:-1])
                    if main_group not in result:
                        result[main_group] = {}
                    result[main_group]['names'] = value
                else:
                    result[attr_name] = value
            else:
                if isinstance(value, dict):
                    # TOML has no null type - omit None-valued keys
                    filtered = {k: v for k, v in value.items() if v is not None}
                    if filtered:
                        result[attr_name] = filtered
                else:
                    result[attr_name] = value

        # Build sorted_result: sections first, then phase groups, then remaining keys
        sorted_result = {}

        # Emit the config sections in a stable order first
        for category in ['reactive', 'emulator']:
            if category in result:
                sorted_result[category] = result[category]

        # Group phase sections
        phase_groups = {}
        other_keys = []
        for key in result.keys():
            if key in sorted_result:
                continue
            if key.endswith('_phases'):
                phase_groups.setdefault(key, []).append(key)
            elif '.' in key:
                main_phase = key.split('.')[0]
                if main_phase.endswith('_phases'):
                    phase_groups.setdefault(main_phase, []).append(key)
                else:
                    other_keys.append(key)
            else:
                other_keys.append(key)

        # Add phase groups
        for phase_type in sorted(phase_groups.keys()):
            phase_keys = phase_groups[phase_type]
            main_key = phase_type
            sub_keys = [k for k in phase_keys if k != main_key]

            if main_key in phase_keys:
                sorted_result[main_key] = result[main_key]

            for sub_key in sorted(sub_keys):
                sorted_result[sub_key] = result[sub_key]

        # Add remaining keys alphabetically
        for key in sorted(other_keys):
            sorted_result[key] = result[key]
        return sorted_result

    def _update_schema_for_new_attrs(self, attr_names):
        """Update configuration schema for new attributes."""
        if not hasattr(self, '_config_schema'):
            self._config_schema = {}
        for attr_name in attr_names:
            if attr_name not in self._config_schema:
                # Auto-detect group based on naming convention
                parts = attr_name.split('_')

                if len(parts) >= 2 and parts[0] in ['equilibrium', 'kinetic']:
                    main_group = '_'.join(parts[:2])  # equilibrium_phases
                    sub_group = parts[2] if len(parts) > 2 else 'default'

                    # Store the nested structure info
                    self._config_schema[attr_name] = {
                        'main_group': main_group,
                        'sub_group': sub_group,
                        'key': '_'.join(parts[3:]) if len(parts) > 3 else parts[2]
                    }
                else:
                    self._config_schema[attr_name] = {
                        'main_group': 'general',
                        'sub_group': None,
                        'key': attr_name
                    }
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MF6RTMConfig':
        """Build a configuration from a nested dictionary.

        Parameters
        ----------
        config_dict : dict
            Nested configuration mapping. The ``reactive`` and ``emulator``
            sections are handled explicitly; remaining sections are flattened.

        Returns
        -------
        MF6RTMConfig
        """
        kwargs = {}
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
            """Flatten a nested dict into ``sep``-joined single-level keys."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Handle reactive section manually and skip flattening it
        if 'reactive' in config_dict:
            reactive_config = config_dict['reactive']
            kwargs['reactive'] = {
                'enabled': reactive_config.get('enabled', True),
                'timing': reactive_config.get('timing', 'all'),
                'tsteps': reactive_config.get('tsteps', []),
                'externalio': reactive_config.get('externalio', False)
            }

        if 'emulator' in config_dict:
            emu_config = config_dict['emulator']
            kwargs['emulator'] = {
                'training_data': emu_config.get('training_data', True),
                'feature_variables': emu_config.get('feature_variables', None),
                'target_variables': emu_config.get('target_variables', None)
            }

        if 'output' in config_dict:
            out_config = config_dict['output']
            kwargs['output'] = {
                'output_format': out_config.get('output_format', 'csv'),
            }

        if 'solver' in config_dict:
            solver_config = config_dict['solver']
            kwargs['solver'] = {
                'min_concentration': solver_config.get('min_concentration', None),
                'no_react_cells': solver_config.get('no_react_cells', None),
            }
            # carry the diffmask threshold through the round-trip when set
            if 'threshold' in solver_config:
                kwargs['solver']['threshold'] = solver_config['threshold']

        # Flatten everything *except* known sections
        remaining_dict = {k: v for k, v in config_dict.items() if k not in ['reactive', 'solver', 'output',
                                                                            # 'emulator'
                                                                            ]}
        flattened = flatten_dict(remaining_dict)

        # Important: remove any reactive_* keys that may be left from TOML
        for k in list(flattened.keys()):
            if k.startswith("reactive_"):
                del flattened[k]

        kwargs.update(flattened)
        return cls(**kwargs)

    @classmethod
    def from_toml_file(cls, filepath: str) -> 'MF6RTMConfig':
        """Load configuration from TOML file.

        Parameters
        ----------
        filepath : str
            Path to TOML configuration file.

        Returns
        -------
        MF6RTMConfig
            New configuration instance loaded from file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = toml.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML format in {filepath}: {e}")

    def save_to_file(self, filepath: str):
        """Save configuration to TOML file.

        Parameters
        ----------
        filepath : str
            Path where TOML file should be saved.
        """
        if os.path.exists(filepath):
            #remove existing file
            os.remove(filepath)
        # Convert configuration to dictionary
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            toml.dump(config_dict, f)
        # print(f"Configuration saved to: {filepath}")

    def __repr__(self):
        """String representation of the configuration."""
        # Get all instance attributes (excluding private/protected ones)
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

        # Format each attribute as key=value
        attr_strs = []
        for key, value in attrs.items():
            if isinstance(value, str):
                attr_strs.append(f"{key}='{value}'")
            else:
                attr_strs.append(f"{key}={value}")

        return f"{self.__class__.__name__}({', '.join(attr_strs)})"

    def __str__(self):
        """Detailed string representation."""
        lines = [f"MF6RTM will run with the following configuration:"]
        lines.append(f"  Reactive: {self.reactive['enabled']}")
        lines.append(f"  Reaction timing: {self.reactive['timing']}")
        lines.append(f"  External files flag: {self.reactive['externalio']}")
        lines.append(f"  Emulator flag: {self.emulator['training_data']}")
        min_conc = self.solver.get('min_concentration')
        lines.append(f"  Min concentration: {min_conc:.2e} mol/L" if min_conc is not None else "  Min concentration: None (no clipping)")
        if self.reactive['timing'] == 'user' and self.reactive['tsteps']:
            lines.append(f"  User-defined time steps ({len(self.reactive['tsteps'])} total):")
            for kper, kstp in sorted(self.reactive['tsteps']):
                lines.append(f"    Period {kper}, Step {kstp}")
        elif self.reactive['timing'] == 'all':
            lines.append("  Reactions calculated at all time steps")

        return '\n'.join(lines)
