# PSSM Dictionary Generator

This script generates Position-Specific Scoring Matrices (PSSMs) from a given FASTA file using PSI-BLAST, and then creates a dictionary where sequences are mapped to their corresponding PSSMs. The dictionary is saved in a serialized pickle format.

## Prerequisites

Before running this script, ensure you have the following installed:

1. **Python 3.9+**: Make sure Python is installed on your system. You can download it from [Python.org](https://www.python.org/downloads/).
2. **BLAST Tools**: This script uses `psiblast`, which is part of the BLAST+ suite of tools. You can download and install it from [NCBI BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).

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

3. **Download Protein Database**:

    You can download the database from the following link:
        [Zenodo: Swiss-Prot Database](https://zenodo.org/records/12788649)
    
    After downloading, ensure that the swissprot database files are placed in the `db` directory or another directory of your choice. If you choose a different directory, update the `DB_PATH` variable in the script accordingly.

    Example folder structure:

    ```
    db/
    ├── swissprot.pdb
    ├── swissprot.phr
    ├── swissprot.pin
    ├── swissprot.pot
    ├── swissprot.psq
    ├── swissprot.ptf
    └── swissprot.pto
    ```

    The Swiss-Prot database for PSI-BLAST was prepared with the following command to create the database in the desired format:
    
    You do not need to run this command!
    ```bash
    makeblastdb -in swissprot.fasta -dbtype prot -out db/swissprot
    ```

    
    


## Usage

To run the script, use the following command:

```bash
python3 create_pssm_dict.py <fasta_file> <output_file>
```

- `<fasta_file>`: Path to the input FASTA file containing the sequences.
- `<output_file>`: Path where the output dictionary (in pickle format) will be saved.

### Example

```bash
python3 create_pssm_dict.py input_sequences.fasta output_pssm_dict.pickle
```

## Script Overview

- **get_seq(file_path)**: Extracts sequence data from the PSSM file.
- **get_pssm_arr(file_path)**: Converts the PSSM data into a NumPy array.
- **run_psiblast(fasta_file, output_folder, db_path)**: Runs PSI-BLAST on each sequence in the FASTA file and saves the resulting PSSMs.
- **Main Script**: Generates PSSMs, reads them into a dictionary, and saves the dictionary as a pickle file.
