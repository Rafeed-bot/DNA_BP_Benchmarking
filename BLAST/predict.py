import subprocess
from Bio import SeqIO
import os
import argparse
import csv
import re

def extract_similarity_info(blast_output):
    e_values = []
    identity_percentages = []
    
    for line in blast_output:
        if "Expect = " in line:
            e_value_match = re.search(r'Expect = ([0-9e\-.]+)', line)
            if e_value_match:
                e_values.append(e_value_match.group(1))
                
        if 'Identities' in line:
            identity_match = re.search(r'Identities = (\d+)\/(\d+)', line)
            if identity_match:
                identity_percentage = "{:.0f}%".format((int(identity_match.group(1)) / int(identity_match.group(2))) * 100)
                identity_percentages.append(identity_percentage)
        
    return e_values, identity_percentages


def is_similar(e_values, identity_percentages, threshold=0.01):
    # Iterate through each e-value and corresponding identity percentage
    for e_value, identity_percentage in zip(e_values, identity_percentages):
        # Check if both criteria are fulfilled
        if float(e_value) < threshold and float(identity_percentage.strip('%')) > 35:
            return 1
    return 0


def run_blast(fasta_file, output_folder, db_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    seq = []
    similarity_info = []
    for s in sequences:
        seq.append(s.seq)
    
    # Run psiblast for each sequence
    for i, sequence in enumerate(sequences):

        temp_fasta_file = os.path.join(output_folder, f"temp_sequence.fasta")
        SeqIO.write(sequence, temp_fasta_file, "fasta")
        
        command = f"blastp -query {temp_fasta_file} -db {db_path} -out out.txt"

        # Run the psiblast command
        subprocess.run(command, shell=True)

        # parse out
        with open("out.txt", "r") as blast_result:
            lines = blast_result.readlines()
            found_alignment = False
            for line in lines:
                if found_alignment: 
                    e_value, identity_percentage = extract_similarity_info(lines)
                    # similarity_info.append(is_similar(e_value, identity_percentage))
                    similarity_info.append(is_similar(e_value, identity_percentage))
                    break
                if "Sequences producing significant alignments" in line:
                    found_alignment = True
        
        # remove temp
        os.remove(temp_fasta_file)
        os.remove("out.txt")

    return seq, similarity_info


def write_to_csv(sequences, targets, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence', 'Target'])  # Write header row
        for sequence, target in zip(sequences, targets):
            writer.writerow([sequence, target])


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("db_path", type=str, help="Path to the db")
    parser.add_argument("fasta_file", type=str, help="Path to the input fasta file.")
    parser.add_argument("save_path", type=str, help="Path to save the csv predictions.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("DB path:", args.db_path)
    print("Fasta file path:", args.fasta_file)
    print("Prediction saving path:", args.save_path)

    fasta_file = args.fasta_file
    save_path = args.save_path
    db_path = args.db_path


    seq, target = run_blast(fasta_file, "BLAST", db_path)

    write_to_csv(seq, target, save_path)