endmainblock  = '''\nEND\n
PRINT
	-reset false
END\n'''

def handle_block(current_items, block_generator, i):
    script = ""
    script += block_generator(current_items, i)
    return script

def get_compound_names(database_file, block = 'SOLUTION_MASTER_SPECIES'):
    species_names = []
    with open(database_file, 'r') as db:
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
    return species_names

def generate_phases_block(phases_dict, i):
    script = ""
    script += f"EQUILIBRIUM_PHASES {i+1}\n"
    for name, values in phases_dict.items():
        saturation_index, amount = values
        script += f"    {name} {saturation_index:.5e} {amount:.5e}\n"
    return script

def generate_solution_block(species_dict, i, temp = 25.0, water = 1.0):
    script = f"SOLUTION {i+1}\n"
    script += f'''   units mol/kgw
    water {water}
    temp {temp}\n'''
    for species, concentration in species_dict.items():
        script += f"    {species} {concentration:.5e}\n"
    script += "\nEND\n"
    return script

def rearrange_copy_blocks(script):
    lines = script.split('\n')
    copy_blocks = []
    end_blocks = []
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

