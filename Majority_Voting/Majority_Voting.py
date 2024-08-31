import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--BLAST', type=str, required=True, help='Path for 2 column csv file containing sequences and prediction (0 or 1) from BLAST')
parser.add_argument('--Local_DPP', type=str, required=True, help='Path for 2 column csv file containing sequences and prediction (0 or 1) from Local-DPP')
parser.add_argument('--LSTM_CNN_Fusion', type=str, required=True, help='Path for 2 column csv file containing sequences and prediction (0 or 1) from LSTM-CNN_Fusion')
parser.add_argument('--Output', type=str, required=True, help='Path for final prediction (using majority voting from BLAST, Local-DPP and LSTM-CNN_Fusion) output csv file')

args = parser.parse_args()
Blast_file, Local_DPP_file, LSTM_CNN_Fusion_file = args.BLAST, args.Local_DPP, args.LSTM_CNN_Fusion
output_file = args.Output

blast_preds = pd.read_csv(Blast_file).values.tolist()
local_preds = pd.read_csv(Local_DPP_file).values.tolist()
lstm_preds = pd.read_csv(LSTM_CNN_Fusion_file).values.tolist()

def find_pred(seq, samples):
    for sample in samples:
        if seq == sample[0]:
            return sample[1]

csv_list = []
for blast_pred in blast_preds:
    pred_labels = []
    pred_labels.append(blast_pred[1])
    pred_labels.append(find_pred(blast_pred[0], local_preds))
    pred_labels.append(find_pred(blast_pred[0], lstm_preds))
    max_integer = max([0, 1], key=pred_labels.count)
    csv_list.append([blast_pred[0], max_integer])
df = pd.DataFrame(csv_list)
df.to_csv(output_file, index=False, header=['sequences', 'predictions'])

