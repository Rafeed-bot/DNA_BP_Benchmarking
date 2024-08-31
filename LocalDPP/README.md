# LocalDPP Protein Sequence Prediction Script

This script implements LocalDPP, predicting labels for protein sequences based on their Position-Specific Scoring Matrices (PSSMs). The trained model has been included for your convenience. The predictions are saved in a CSV file.

## Prerequisites

Before running this script, ensure you have the following installed:

1. **Python 3.9+**: Make sure Python is installed on your system. You can download it from [Python.org](https://www.python.org/downloads/).
2. **Required Python packages**: Listed in the `requirements.txt` file.
3. Make sure you have generated the PSSM dictionary from PSSM_Construction folder.

## Installation

1. **Clone the repository** (if applicable):
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the script, use the following command:

```bash
python3 predict.py <pssm_dict> <fasta_file> <save_path>
```

### Arguments:

- `<pssm_dict>`: Path to the PSSM dictionary file (in pickle format).
- `<fasta_file>`: Path to the FASTA file containing the sequences to be predicted.
- `<save_path>`: Path to save the resulting CSV file with predictions.

### Example

```bash
python3 predict.py pssm_dict.pickle input.fasta predictions.csv
```

This command will:

1. Load the PSSM dictionary from `pssm_dict.pickle`.
2. Read the sequences from `input.fasta`.
3. Generate feature vectors for the sequences using the PSSMs.
4. Load the pre-trained model from `LocalDPP/model.pkl`.
5. Predict labels for the sequences and save the results to `predictions.csv`.

### Example Output

The resulting CSV file will have the following structure:

| Sequences         | Predicted Labels |
|-------------------|------------------|
| **Sequence_1**    | 1                |
| **Sequence_2**    | 0                |
| ...               | ...              |
