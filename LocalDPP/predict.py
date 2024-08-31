# args: prediction csv file path, PSSM dictionary path (if required), prediction csv file save path

import argparse
import numpy as np
import pickle
import pandas as pd
from Bio import SeqIO

MODEL_PATH = 'LocalDPP/model.pkl'

# Hyperparameters
N = 3
LAMDBA = 1

def normalize_matrix(matrix):
    normalized = []

    for row in matrix:
        normalized_row = []
        squared_val = 0
        row_mean = np.mean(row)

        for value in row:
            numerator = value - row_mean
            numerator_squared = numerator ** 2

            normalized_row.append(numerator)
            squared_val += numerator_squared

        denominator = np.sqrt((1/20) * squared_val)

        # Handle division by 0 
        if denominator != 0:
            normalized_row = np.array(normalized_row) / denominator
        else:
            normalized_row = np.zeros_like(normalized_row)
        # normalized_row = np.array(normalized_row) / denominator

        normalized.append(normalized_row)

    return np.array(normalized)


def row_fragment_matrix(matrix, n):
    submatrices = np.array_split(matrix, n, axis=0)
    return submatrices


# Equation (4)
def part_one(matrix, L):
    column_sums = np.sum(matrix, axis=0)
    return column_sums / (L/N)


# Equation (5)
def part_two(matrix, L):
    final = [None] * len(matrix[0]) 
    for i in range(len(matrix[0])):
        for j in range(len(matrix) - 1):
            final[i] = (matrix[j][i] - matrix[j + 1][i]) ** 2

    denominator = 1 / ((L / N) - 1)
    final = np.array(final) / denominator
    return final 


def generate_feature_vector(pssm):
    length = len(pssm)

    normalized = normalize_matrix(pssm)
    fragments = row_fragment_matrix(normalized, N)
    
    feature_vector = np.array([])
    for fragment in fragments:
        p1 = part_one(fragment, length)
        p2 = part_two(fragment, length)

        combined = np.concatenate((p1, p2))
        feature_vector = np.concatenate((feature_vector, combined), axis=None)
    return feature_vector


def write_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def main(fasta_file_path, dic, file_path, trained_model):
    
    all_seq = read_fasta(fasta_file_path)
    feature_vectors = []
    seq = []

    for value in all_seq:
        if value in dic:
            feature_vectors.append(generate_feature_vector(dic[value]))
            seq.append(value)

    feature_vectors = np.array(feature_vectors)

    # load model from pickle file
    with open(trained_model, 'rb') as file:  
        model = pickle.load(file)

    # evaluate model 
    y_predict = model.predict_proba(feature_vectors)
    positive_proba = np.array(y_predict)[:, 1]  # Assuming positive class is the second class
    predictions = (positive_proba >= 0.5).astype(int)

    data = {
        'Sequences': seq,
        'Predicted Labels': predictions
    }
    
    write_to_csv(data, file_path)
    print("Predictions saved to:", file_path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("pssm_dict", type=str, help="Path to the PSSM dictionary file")
    parser.add_argument("fasta_file", type=str, help="Path to the fasta file.")
    parser.add_argument("save_path", type=str, help="Path to save the csv predictions.")


    return parser.parse_args()


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


if __name__ == '__main__':
    args = parse_arguments()

    print("PSSM dictionary path:", args.pssm_dict)
    print("Fasta file path:", args.fasta_file)
    print("Prdiction saving path:", args.save_path)
    

    fasta_file = args.fasta_file
    dic = load_pickle_file(args.pssm_dict)
    save_path = args.save_path
    model_path = MODEL_PATH

    main(fasta_file, dic, save_path, model_path)

