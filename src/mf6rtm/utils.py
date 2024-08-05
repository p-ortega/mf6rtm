import platform
import os
import shutil
import pandas

# global variables
endmainblock = '''\nPRINT
    -reset false
END\n'''


def solution_csv_to_dict(csv_file, header=True):
    """Read a solution CSV file and convert it to a dictionary
    Parameters
    ----------
    csv_file : str
        The path to the solution CSV file.
    header : bool, optional
        Whether the CSV file has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with the first column as keys and the remaining columns as values.
    """
    # Read the CSV file and convert it to a dictionary using first row as keys and columns as value (array of shape ncol)
    import csv
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        # skip header assuming first line is header
        if header:
            next(reader)

        data = {rows[0]: [float(i) for i in rows[1:]] for rows in reader if not rows[0].startswith('#')}
        # data = {rows[0]: rows[1:] for rows in reader if rows[0].startswith('#') == False}

        # for key, value in data.items():
        #     data[key] = [float(i) for i in value]
    return data


def kinetics_df_to_dict(data, header=True):
    """Read a kinetics CSV file and convert it to a dictionary
    Parameters
    ----------
    csv_file : str
        The path to the kinetics CSV file.
    header : bool, optional
        Whether the CSV file has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with the first column as keys and the remaining columns as values.
    """
    dic = {}
    # data.set_index(data.columns[0], inplace=True)
    par_cols = [col for col in data.columns if col.startswith('par')]
    for key in data.index:
        parms = [item for item in data.loc[key, par_cols] if not pandas.isna(item)]
        # print(parms)
        dic[key] = [item for item in data.loc[key] if item not in parms and not pandas.isna(item)]
        dic[key].append(parms)
    return dic


def solution_df_to_dict(data, header=True):
    """Convert a pandas DataFrame to a dictionary
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to convert.
    header : bool, optional
        Whether the DataFrame has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with the first column as keys and the remaining columns as values.
    """
    data = data.T.to_dict('list')
    for key, value in data.items():
        data[key] = [float(i) for i in value]
    return data


def equilibrium_phases_csv_to_dict(csv_file, header=True):
    """Read an equilibrium phases CSV file and convert it to a dictionary
    Parameters
    ----------
    csv_file : str
        The path to the equilibrium phases CSV file.
    header : bool, optional
        Whether the CSV file has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with phase names as keys and lists of saturation indices and amounts as values.
    """
    import csv
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        # skip header assuming first line is header
        if header:
            next(reader)
        data = {}
        for row in reader:
            if row[0].startswith('#'):
                continue
            if int(row[-1]) not in data:
                # data[row[0]] = [[float(row[1]), float(row[2])]]
                data[int(row[-1])] = {row[0]: [float(row[1]), float(row[2])]}
            else:
                # data[int(row[-1])] # append {row[0]: [float(row[1]), float(row[2])]} to the existing nested dictionary
                data[int(row[-1])][row[0]] = [float(row[1]), float(row[2])]
    return data


def surfaces_csv_to_dict(csv_file, header=True):
    """Read an equilibrium phases CSV file and convert it to a dictionary
    Parameters
    ----------
    csv_file : str
        The path to the equilibrium phases CSV file.
    header : bool, optional
        Whether the CSV file has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with phase names as keys and lists of saturation indices and amounts as values.
    """
    import csv
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        # skip header assuming first line is header
        if header:
            next(reader)
        data = {}
        for row in reader:
            if row[0].startswith('#'):
                continue
            if int(row[-1]) not in data:
                # data[row[0]] = [[float(row[1]), float(row[2])]]
                data[int(row[-1])] = {row[0]: [i for i in row[1:-1]]}
            else:
                # data[int(row[-1])] # append {row[0]: [float(row[1]), float(row[2])]} to the existing nested dictionary
                data[int(row[-1])][row[0]] = [i for i in row[1:-1]]
    return data


def kinetics_phases_csv_to_dict(csv_file, header=True):
    """Read an equilibrium phases CSV file and convert it to a dictionary
    Parameters
    ----------
    csv_file : str
        The path to the equilibrium phases CSV file.
    header : bool, optional
        Whether the CSV file has a header. The default is True.
    Returns
    -------
    data : dict
        A dictionary with phase names as keys and lists of saturation indices and amounts as values.
    """
    import csv
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        # skip header assuming first line is header
        if header:
            cols = next(reader)
            # print(cols)
        data = {}
        for row in reader:
            if row[0].startswith('#'):
                continue
            rowcleaned = [i for i in row if i != '']
            if int(rowcleaned[-1]) not in data:
                # data[row[0]] = [[float(row[1]), float(row[2])]]
                data[int(rowcleaned[-1])] = {rowcleaned[0]: [float(rowcleaned[1])]}
                data[int(rowcleaned[-1])][rowcleaned[0]].append([float(i) for i in rowcleaned[2:-1]])
            else:
                data[int(rowcleaned[-1])][rowcleaned[0]] = [float(rowcleaned[1])]
                data[int(rowcleaned[-1])][rowcleaned[0]].append([float(i) for i in rowcleaned[2:-1]])
                # [float(i) for i in rowcleaned[1:-1]]
    return data


def handle_block(current_items, block_generator, i, *args, **kwargs):
    """Generate a block for a PHREEQC input script if the current items are not empty
    Parameters
    ----------
    current_items : list
        A list of items to include in the block.
    block_generator : function
        A function that generates the block.
    i : int
        The block number.
    Returns
    -------
    script : str
        The block as a string.
    """
    # temp = kwargs.get('temp')  # Safely get 'temp' if it exists, else returns None
    # water = kwargs.get('water')

    script = ""
    script += block_generator(current_items, i, *args, **kwargs)
    return script


def get_compound_names(database_file, block='SOLUTION_MASTER_SPECIES'):
    """Get a list of compound names from a PHREEQC database file
    Parameters
    ----------
    database_file : str
        The path to the PHREEQC database file.
    block : str, optional
        The keyword for the block containing the compound names. The default is 'SOLUTION_MASTER_SPECIES'.
    Returns
    -------
    compound_names : list
        A list of compound names.
    """
    species_names = []
    with open(database_file, 'r', errors='replace') as db:
        lines = db.readlines()
        in_block = False
        for line in lines:
            if block.upper() in line:
                in_block = True
            elif in_block and line.strip().isupper() and len(line.strip()) > 1 and '_' in line.strip():  # Stop when encountering the next keyword
                in_block = False
            elif in_block:
                if line.strip() and not line.startswith('#') and line.split()[0][0].isupper():  # Ignore empty lines and comments
                    species = line.split()[0]  # The species name is the first word on the line
                    species_names.append(species)
                if line.strip() and not line.startswith('#') and line.split()[0][0].isupper() and block.startswith('EXCHANGE'):
                    # Ignore empty lines and comments
                    species = line.split()[-1]  # The exchange species are the last word on the line
                    species_names.append(species)
    return species_names


def generate_exchange_block(exchange_dict, i, equilibrate_solutions=[]):
    """Generate an EXCHANGE block for PHREEQC input script
    Parameters
    ----------
    exchange_dict : dict
        A dictionary with species names as keys and exchange concentrations as values.
    i : int
        The block number.
    Returns
    -------
    script : str
        The EXCHANGE block as a string.
    """
    script = f"EXCHANGE {i+1}\n"
    for species, conc in exchange_dict.items():
        script += f"    {species} {conc:.5e}\n"
    if len(equilibrate_solutions) > 0:
        script += f"    -equilibrate {equilibrate_solutions[i]}"
    else:
        script += f"    -equilibrate {1}"
    script += "\nEND\n"
    return script


def generate_surface_block(surface_dict, i, options=[]):
    """Generate a SURFACE block for PHREEQC input script
    Parameters
    ----------
    surface_dict : dict
        A dictionary with surface names as keys and lists of site densities and site densities as values.
    i : int
        The block number.
    Returns
    -------
    script : str
        The SURFACE block as a string.
    """
    script = f"SURFACE {i+1}\n"
    for name, values in surface_dict.items():
        script += f"    {name}"
        script += "    "+' '.join(f"{v}" for v in values)+"\n"
        script += f"    -equilibrate {1}\n"  # TODO: make equilibrate a parameter from eq_solutions
        if len(options) > 0:
            for i in range(len(options)):
                script += f"    -{options[i]}\n"
            # script += f"    -{options[i]}\n"
        # script += f"    -no_edl\n"
    script += "END\n"
    return script


def generate_kinetics_block(kinetics_dict, i):
    """Generate a KINETICS block for PHREEQC input script
    Parameters
    ----------
    kinetics_dict : dict
        A dictionary with species names as keys and lists of rate constants and exponents as values.
    i : int
        The block number.
    Returns
    -------
    script : str
        The KINETICS block as a string.
    """
    script = f"KINETICS {i+1}\n"
    options = ["m0", "parms", "formula"]
    for species, values in kinetics_dict.items():
        script += f"    {species}\n"
        for k in range(len(values)):
            if isinstance(values[k], list):
                script += f"        -{options[k]} " + ' '.join(f"{parm:.5e}" for parm in values[k])+"\n"
            elif isinstance(values[k], str):
                script += f"        -{options[k]} {values[k]}\n"
            else:
                script += f"        -{options[k]} {values[k]:.5e}\n"
    script += "\nEND\n"
    return script


def generate_phases_block(phases_dict, i):
    """Generate an EQUILIBRIUM_PHASES block for PHREEQC input script
    Parameters
    ----------
    phases_dict : dict
        A dictionary with phase names as keys and lists of saturation indices and amounts as values.
    i : int
        The block number.
    Returns
    -------
    script : str
        The EQUILIBRIUM_PHASES block as a string.
    """
    script = ""
    script += f"EQUILIBRIUM_PHASES {i+1}\n"
    for name, values in phases_dict.items():
        saturation_index, amount = values
        script += f"    {name} {saturation_index:.5e} {amount:.5e}\n"
    script += "\nEND\n"
    return script


def generate_solution_block(species_dict, i, temp=25.0, water=1.0):
    """Generate a SOLUTION block for PHREEQC input script
    Parameters
    ----------
    species_dict : dict
        A dictionary with species names as keys and concentrations as values.
    i : int
        The solution number.
    temp : float, optional
        The temperature of the solution in degrees Celsius. The default is 25.0.
    water : float, optional
        The mass of water in kg. The default is 1.0.
    Returns
    -------
    script : str
        The SOLUTION block as a string.
    """
    if isinstance(temp, (int, float)):
        t = f"{temp:.1f}"
    elif isinstance(temp, list):
        t = f"{temp[i]}"
    script = f"SOLUTION {i+1}\n"
    script += f'''   units mol/kgw
    water {water}
    temp {t}\n'''
    for species, concentration in species_dict.items():
        script += f"    {species} {concentration:.5e}\n"
    script += "\nEND\n"
    return script


def rearrange_copy_blocks(script):
    # Split the script into lines
    lines = script.split('\n')
    copy_blocks = []
    # end_blocks = []
    other_blocks = []

    # Separate the lines into COPY blocks, END blocks, and other blocks
    for line in lines:
        if line.startswith('COPY'):
            copy_blocks.append(line)
        else:
            other_blocks.append(line)

    # Combine the blocks, putting the COPY blocks at the end and avoiding consecutive END blocks
    rearranged_script = []
    for block in other_blocks + copy_blocks:
        rearranged_script.append(block)

    # Join the lines back together into a single script string
    rearranged_script = '\n'.join(rearranged_script)

    return rearranged_script


def prep_bins(dest_path, src_path=os.path.join('bin'),  get_only=[]):
    """Copy executables from the source path to the destination path
    """

    if "linux" in platform.platform().lower():
        bin_path = os.path.join(src_path, "linux")
    elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
        bin_path = os.path.join(src_path, "mac")
    else:
        bin_path = os.path.join(src_path, "win")
    files = os.listdir(bin_path)
    if len(get_only) > 0:
        files = [f for f in files if f.split(".")[0] in get_only]

    for f in files:
        if os.path.exists(os.path.join(dest_path, f)):
            try:
                os.remove(os.path.join(dest_path, f))
            except IOError:
                continue
        shutil.copy2(os.path.join(bin_path, f), os.path.join(dest_path, f))
    return
