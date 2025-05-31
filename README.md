# CaseFlo


## Setup Instructions

### 1. Clone git repository

```bash
git clone https://github.com/jr4fs/CaseFlo.git
cd CaseFlo
```

### 1. Create a Conda or venv Environment

```bash
conda create -n caseflo python=3.10 -y
conda activate caseflo
```

```bash
python3 -m venv caseflo
source caseflo/bin/activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

if using venv... 
```bash
pip install -r requirements_venv.txt
python -m spacy download en_core_web_lg
```

### 2. Example Usage
#### Anonymizer
```bash
python src/anonymizer.py --file input.csv --output anonymized.csv --column <name of column with text to anonymize>
```
