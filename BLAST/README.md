# BLAST Protein Sequence Prediction Script

This script runs a BLAST search against a database of positive sequences and checks if any similar sequences have an e-value < 0.01 and a similarity score > 35%. It outputs a CSV file indicating whether each input sequence meets the similarity criteria.


## Prerequisites

Before running this script, ensure you have the following installed:

1. **Python 3.9+**: Make sure Python is installed on your system. You can download it from [Python.org](https://www.python.org/downloads/).
2. **Required Python packages**: Listed in the `requirements.txt` file.
3. **BLAST Tools**: This script uses `blastp`, which is part of the BLAST+ suite of tools. You can download and install it from [NCBI BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).


## Input Requirements
- FASTA file: The input file should be in FASTA format containing one or more sequences.
- BLAST Database: The database should be formatted for BLAST and contain files like .pdb, .phr, .pin, etc.
- Database provided can be found under Models.zip, BLAST_db/positive


Example folder structure:

    ```
    BLAST_db/
    ├── positive.pdb
    ├── positive.phr
    ├── positive.pin
    ├── positive.pot
    ├── positive.psq
    ├── positive.ptf
    └── positive.pto
    ```

You would just need to call the script with BLAST_db/positive as your first argument.

You can create a BLAST database using the makeblastdb command or download one from public sources.

## Usage

To run the script, use the following command:

```bash
python3 predict.py <db_path> <input_fasta> <output_file>
```

- `<db_path>`: Path to folder containing DB files
- `<input_fasta>`: Path to the input FASTA file containing the sequences.
- `<output_file>`: Path where the output CSV file will be saved.saved.

### Example

```bash
python3 predict.py ../Models/BLAST_db/positive input_sequences.fasta results.csv
```


## Output
The script produces a CSV file with two columns:

- Sequence: The original input sequence.
- Target: A binary value (1 or 0) indicating whether the sequence meets the similarity criteria.
