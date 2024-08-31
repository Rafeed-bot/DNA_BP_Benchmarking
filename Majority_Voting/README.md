# Majority Voting

This script performs majority voting on the predictions made by BLAST, Local-DPP and LSTM-CNN_Fusion.

## Prerequisites

You need to have **Python 3.9+** and **Pandas** installed

## Usage

To run the script, use the following command:

```bash
python Majority_Voting.py --BLAST <Blast_csv> --Local_DPP <Local-DPP_csv> --LSTM_CNN_Fusion <LSTM-CNN_Fusion_csv> --Output <predictions_csv>
```

### Arguments:

- `<Blast_csv>`: Path for two column csv file containing sequences and prediction (0 or 1) from BLAST.
- `<Local-DPP_csv>`: Path for two column csv file containing sequences and prediction (0 or 1) from Local-DPP.
- `<LSTM-CNN_Fusion_csv>`: Path for two column csv file containing sequences and prediction (0 or 1) from LSTM-CNN_Fusion.
- `<predictions_csv>`: Path for final prediction (using majority voting from BLAST, Local-DPP and LSTM-CNN_Fusion) output csv file.

### Example

```bash
python Majority_Voting.py --BLAST Blast.csv --Local_DPP LocalDPP.csv --LSTM_CNN_Fusion LSTM_CNN_Fusion.csv --Output predictions.csv
```

### Example Output

The resulting CSV file will have the following structure:

| Sequences         | Predicted Labels |
|-------------------|------------------|
| **Sequence_1**    | 1                |
| **Sequence_2**    | 0                |
| ...               | ...              |
