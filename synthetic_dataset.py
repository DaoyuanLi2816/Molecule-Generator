import random
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors


def consecutive_hydroxyls(chain):
    """
    Check if the chain contains five or more consecutive hydroxyl groups.

    Args:
        chain (list of str): The carbon chain with possible hydroxyl groups. 

    Returns:
        bool: True if there are five or more consecutive hydroxyl groups, False otherwise.
    """
    count = 0
    max_count = 0
    i = 0
    while i < len(chain) - 1:
        if chain[i] == 'C' and chain[i + 1] == '(':
            count += 1
            max_count = max(max_count, count)
        elif chain[i] == 'C' and chain[i + 1] == 'C':
            count = 0
        i += 1
    return max_count >= 5


def generate_molecule(min_carbon, max_carbon):
    """
    Generate a random molecule within a specified range of carbon atoms.

    Args:
        min_carbon (int): Minimum number of carbon atoms.
        max_carbon (int): Maximum number of carbon atoms.

    Returns:
        str: The SMILES string of the generated molecule.
    """
    while True:
        num_carbon = random.randint(min_carbon, max_carbon)
        carbon_chain = ['C' for _ in range(num_carbon)]

        # Randomly generate hydroxyl groups
        num_hydroxyl = random.randint(1, num_carbon)
        hydroxyl_positions = random.sample(range(num_carbon), num_hydroxyl)
        for pos in hydroxyl_positions:
            carbon_chain[pos] += '(O)'

        if consecutive_hydroxyls(carbon_chain):
            continue

        # Randomly generate branches
        num_branches = random.randint(0, num_carbon // 2)
        branch_positions = random.sample(range(num_carbon), num_branches)
        for pos in branch_positions:
            branch_length = random.randint(1, 10)
            branch = 'C' * branch_length
            carbon_chain[pos] += '(' + branch + ')'

        smiles = ''.join(carbon_chain)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return smiles


def save_molecules_to_csv(filename, molecules):
    """
    Save the generated SMILES strings to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        molecules (list of str): The list of SMILES strings.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['SMILES'])  # Write the header

        for smiles in molecules:
            csv_writer.writerow([smiles])  # Write each row

    print(f'Successfully saved {len(molecules)} generated molecules to {filename}')


# Generate 100,000 unique molecules
generated_molecules = []
num_molecules = 100000

for _ in range(num_molecules):
    smiles = generate_molecule(min_carbon=12, max_carbon=20)
    generated_molecules.append(smiles)

generated_molecules = list(set(generated_molecules))

# Save the generated molecules to a CSV file
save_molecules_to_csv('molecules.csv', generated_molecules)
