# CaseFlo


## 📦 Setup Instructions

### 1. Create a Conda Environment

```bash
conda create -n caseflo python=3.10 -y
conda activate caseflo

pip install -r requirements.txt
python -m spacy download en_core_web_lg

python anonymize_text.py --file input.txt --output anonymized.txt
