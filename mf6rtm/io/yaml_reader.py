"""Helpers to reconstruct a YAMLPhreeqcRM instance from a YAML file."""
import warnings

import numpy as np
import yaml

try:
    from phreeqcrm import yamlphreeqcrm
except ImportError:
    warnings.warn("phreeqcrm not found.", ImportWarning)

def load_yaml_to_phreeqcrm(yaml_file_path):
    """Load a YAML file and reconstruct a YAMLPhreeqcRM instance.

    Parameters
    ----------
    yaml_file_path : str
        Path to the YAML file.

    Returns
    -------
    YAMLPhreeqcRM
        Reconstructed instance.
    """

    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Create a new YAMLPhreeqcRM instance
    yrm = yamlphreeqcrm.YAMLPhreeqcRM()

    # Method mapping dictionary
    method_mapping = {
        'SetGridCellCount': lambda item: yrm.YAMLSetGridCellCount(item['count']),
        'ThreadCount': lambda item: yrm.YAMLThreadCount(item['nthreads']),
        'SetComponentH2O': lambda item: yrm.YAMLSetComponentH2O(item['tf']),
        'UseSolutionDensityVolume': lambda item: yrm.YAMLUseSolutionDensityVolume(item['tf']),
        'SetFilePrefix': lambda item: yrm.YAMLSetFilePrefix(item['prefix']),
        'OpenFiles': lambda item: yrm.YAMLOpenFiles(),
        'SetErrorHandlerMode': lambda item: yrm.YAMLSetErrorHandlerMode(item['mode']),
        'SetRebalanceFraction': lambda item: yrm.YAMLSetRebalanceFraction(item['f']),
        'SetRebalanceByCell': lambda item: yrm.YAMLSetRebalanceByCell(item['tf']),
        'SetPartitionUZSolids': lambda item: yrm.YAMLSetPartitionUZSolids(item['tf']),
        'SetUnitsSolution': lambda item: yrm.YAMLSetUnitsSolution(item['option']),
        'SetUnitsPPassemblage': lambda item: yrm.YAMLSetUnitsPPassemblage(item['option']),
        'SetUnitsExchange': lambda item: yrm.YAMLSetUnitsExchange(item['option']),
        'SetUnitsSurface': lambda item: yrm.YAMLSetUnitsSurface(item['option']),
        'SetUnitsGasPhase': lambda item: yrm.YAMLSetUnitsGasPhase(item['option']),
        'SetUnitsSSassemblage': lambda item: yrm.YAMLSetUnitsSSassemblage(item['option']),
        'SetUnitsKinetics': lambda item: yrm.YAMLSetUnitsKinetics(item['option']),
        'SetPorosity': lambda item: yrm.YAMLSetPorosity(item['por']),
        'SetPrintChemistryMask': lambda item: yrm.YAMLSetPrintChemistryMask(item['cell_mask']),
        'SetPrintChemistryOn': lambda item: yrm.YAMLSetPrintChemistryOn(
            item['workers'], item['initial_phreeqc'], item['utility']
        ),
        'SetRepresentativeVolume': lambda item: yrm.YAMLSetRepresentativeVolume(item['rv']),
        'LoadDatabase': lambda item: yrm.YAMLLoadDatabase(item['database']),
        'RunFile': lambda item: yrm.YAMLRunFile(
            item['workers'], item['initial_phreeqc'], item['utility'], item['chemistry_name']
        ),
        'RunString': lambda item: yrm.YAMLRunString(
            item['workers'], item['initial_phreeqc'], item['utility'], item['input_string']
        ),
        'AddOutputVars': lambda item: yrm.YAMLAddOutputVars(item['option'], item['definition']),
        'FindComponents': lambda item: yrm.YAMLFindComponents(),
        'InitialPhreeqc2Module': lambda item: yrm.YAMLInitialPhreeqc2Module(item['ic']),
        'RunCells': lambda item: yrm.YAMLRunCells(),
        'SetTime': lambda item: yrm.YAMLSetTime(item['time']),
    }
    not_to_read = ['RunFile', 'RunString', 'AddOutputVars', 'FindComponents', 'RunCells', 'SetTime']
    # Process each item in the YAML data
    for item in yaml_data:
        key = item.get('key')
        if key in method_mapping:
            if key == 'InitialPhreeqc2Module':
                # Handle the initial conditions separately
                ic1 = np.array(item.get('ic', []))
                continue
            elif key in not_to_read:
                continue
            try:
                method_mapping[key](item)
                # print(f"✓ Processed: {key}")
            except Exception as e:
                print(f"Error processing {key}: {e}")
        else:
            print(f"Unknown key: {key}")

    return yrm, ic1
