# Usage: python predict.py <fasta_file_path> <pssm_dict_path> <output_csv_path>

import sys
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from Bio import SeqIO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnLstmModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(CnnLstmModel, self).__init__()

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(32 * 174 * 4 + hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, lstm_input, cnn_input):
        cnn_output = self.cnn(cnn_input)
        lstm_output, _ = self.lstm(lstm_input)

        cnn_output = cnn_output.view(cnn_output.size(0), -1)

        # Concatenate LSTM and CNN output
        combined_output = torch.cat((lstm_output[:, -1, :], cnn_output), dim=1)

        fc_output = self.linear(combined_output)

        return fc_output


def main(fasta_file_path, pssm_dict_path, trained_model_path, save_path):
    global device

    # Load the dictionary from the pickle file
    with open(pssm_dict_path, 'rb') as file:
        pssm_dict = pickle.load(file)

    # Define hyperparameters
    embedding_dim = 20
    hidden_size = 128

    # Create an instance of the model
    model = CnnLstmModel(embedding_dim, hidden_size).to(device)

    # Load the saved model
    if device == 'cuda':
      model.load_state_dict(torch.load(trained_model_path))
    else:
      model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    

    def read_seq_onehot(sequences):
        result = np.zeros((len(sequences), 700, 20))
        naiive_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        for j, seq in enumerate(sequences):
            if len(seq) > 700:
                seq = seq[:700]
            for i, val in enumerate(seq):
                if val not in naiive_amino_acids:
                    result[j][i] = np.array([0] * 20)
            else:
                index = naiive_amino_acids.index(val)
                result[j][i][index] = 1
        return result

    # Create empty list
    test_pssm_list = []
    sequences_with_pssm_matrix = []

    # Load prediction data
    # Read fasta file
    sequences = []
    with open(fasta_file_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequences.append(str(record.seq))

    for i in range(len(sequences)):
        sequence = sequences[i]
        try:
            pssm_matrix = pssm_dict[sequence]
        except KeyError:
            print(f"Sequence {sequence} not found in the PSSM dictionary.")
            continue
        pssm_matrix = torch.tensor(pssm_matrix, dtype=torch.float)
        if pssm_matrix.shape[0] > 700:
            pssm_matrix = pssm_matrix[:700, :]
        elif pssm_matrix.shape[0] < 700:
            pssm_matrix = F.pad(
                pssm_matrix, (0, 0, 0, 700 - pssm_matrix.shape[0]))

        # Append the PSSM matrix to the list
        test_pssm_list.append(pssm_matrix)

        # Append DNA sequences with PSSM matrix to the list
        sequences_with_pssm_matrix.append(sequence)

    # Set test_pssm_list as torch
    test_pssm_list = torch.stack(test_pssm_list)

    # Create One-hot sequences
    test_one_hot_sequences = read_seq_onehot(sequences_with_pssm_matrix)
    test_one_hot_sequences = torch.from_numpy(test_one_hot_sequences).float()

    test_inputs = torch.cat((test_one_hot_sequences, test_pssm_list), dim=1)

    test_dataset = TensorDataset(test_inputs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()

    # Make predictions
    all_predictions = []
    with torch.no_grad():
        for inputs in test_inputs:
            # inputs = torch.stack(inputs)  # Convert list of tensors to a single tensor
            inputs = inputs.unsqueeze(0)
            lstm_input = inputs[:, :700, :]
            cnn_input = inputs[:, 700:1400, :]

            # Forward pass
            cnn_input = cnn_input.unsqueeze(1)
            outputs = model(lstm_input.to(device), cnn_input.to(device))

            # Calculate predicted labels
            _, predicted = torch.max(outputs, 1)

            # Collect predicted labels and true labels
            all_predictions.extend(predicted.tolist())

    # Save results
    prediction_result = pd.DataFrame({
        'Sequences': sequences_with_pssm_matrix,
        'Predicted Labels': all_predictions
    })
    prediction_result.to_csv(save_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <fasta_file_path> <pssm_dict_path> <output_csv_path>")
        sys.exit(1)

    fasta_file_path = sys.argv[1]
    pssm_dict_path = sys.argv[2]
    trained_model_path = 'model.pt'
    save_path = sys.argv[3]

    main(fasta_file_path, pssm_dict_path, trained_model_path, save_path)
