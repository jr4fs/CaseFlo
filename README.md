# CaseFlo


## Setup Instructions

### 1. Create a Conda Environment

```bash
conda create -n caseflo python=3.10 -y
conda activate caseflo
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
### 2. Example Usage
```bash
python src/anonymizer.py --file input.csv --output anonymized.csv
```