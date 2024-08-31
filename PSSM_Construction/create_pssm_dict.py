import numpy as np
import os
import sys
import pickle
from Bio import SeqIO
import subprocess

DB_PATH = "db/swissprot"
PSSM_FOLDER = "PSSMs"

def get_seq(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = ""
    for line in lines:
        if line.startswith(' '):
            values = line.split()[1]
            data += values
    # Remove last empty list
    data = data[1:-6]
    return data

def get_pssm_arr(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = []
    for line in lines:
        if line.startswith(' '):
            values = line.split()[2:22]
            if all(value.replace('.', '', 1).isdigit() or value.startswith('-') and value[1:].replace('.', '', 1).isdigit() for value in values):
                data.append(values)
    # Remove last empty list
    data = data[0:-1]
    pssm_array = np.array(data, dtype=float)
    return pssm_array

def run_psiblast(fasta_file, output_folder, db_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))

    # Run psiblast for each sequence
    for i, sequence in enumerate(sequences):
        # Define the output filename for the PSSM
        output_file = os.path.join(output_folder, f"{i+1}.pssm")

        # Create a temporary FASTA file for the current sequence
        temp_fasta_file = os.path.join(output_folder, f"temp_sequence.fasta")
        SeqIO.write(sequence, temp_fasta_file, "fasta")

        # Construct the psiblast command for the current sequence
        command = f"psiblast -query {temp_fasta_file} -db {db_path} -evalue 0.001 -num_iterations 3 -out_ascii_pssm {output_file}"

        # Run the psiblast command
        subprocess.run(command, shell=True)

        # Remove the temporary FASTA file
        os.remove(temp_fasta_file)

if __name__ == "__main__":
    try:

        fasta_file = sys.argv[1]
        output_file = sys.argv[2]
    except IndexError:
        print("Usage: python3 create_pssm_dict.py <fasta_file> <output_file>")
        sys.exit(1)

    print("Generating PSSMs")
    run_psiblast(fasta_file, PSSM_FOLDER, DB_PATH)

    train_pssm_files = [f for f in os.listdir(PSSM_FOLDER) if f.endswith(".pssm")]
    dict = {}

    print(f"Reading from {PSSM_FOLDER}")

    for pssm_file in train_pssm_files:
        pssm_path = os.path.join(PSSM_FOLDER, pssm_file)

        seq = get_seq(pssm_path)
        pssm = get_pssm_arr(pssm_path)

        dict[seq] = pssm
    
    with open(output_file, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {output_file}")