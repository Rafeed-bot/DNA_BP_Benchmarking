# LSTM-CNN_Fusion Protein Sequence Prediction Script

This script implements the LSTM-CNN Fusion model, where the CNN extracts features from the PSSMs while the LSTM simultaneously learns information from the original protein sequences. For your convenience, the trained model is included. Predictions are saved in a CSV file, where the first column contains the sequences and the second column contains the predicted labels (0 for non-DBPs and 1 for DBPs).

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
python predict.py <fasta_file_path> <pssm_dict_path> <trained_model_path> <output_csv_path>
```

### Arguments:

- `<fasta_file_path>`: Path to the FASTA file containing the sequences to be predicted.
- `<pssm_dict_path>`: Path to the PSSM dictionary file (in pickle format).
- `<trained_model_path>`: Path to the trained model.
- `<output_csv_path>`: Path to save the resulting CSV file with predictions.

### Example

```bash
python predict.py input.fasta pssm_dict.pickle ../Models/LocalDPP_model.pt predictions.csv
```

This command will:

1. Load the PSSM dictionary from `pssm_dict.pickle`.
2. Read the sequences from `input.fasta`.
3. Load the pre-trained model from `../Models/LocalDPP_model.pkl`.
4. Predict labels for the sequences and save the results to `predictions.csv`.

### Example Output

The resulting CSV file will have the following structure:

| Sequences      | Predicted Labels |
| -------------- | ---------------- |
| **Sequence_1** | 1                |
| **Sequence_2** | 0                |
| ...            | ...              |
