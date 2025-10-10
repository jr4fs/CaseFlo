import re
import argparse
import pandas as pd

import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---- tqdm (progress bars) with safe fallback ----
try:
    from tqdm import tqdm
    from tqdm.contrib import tzip
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    class tqdm:  # fallback no-op
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable or []
        def __iter__(self):
            for x in self.iterable:
                yield x
        def update(self, *args, **kwargs): pass
        def close(self): pass
        @staticmethod
        def write(msg): print(msg)
    def tzip(*args, **kwargs):
        return zip(*args)

# -----------------------------
# Domain allow/deny specifics
# -----------------------------
ALLOWLIST_TERMS = {"yp", "hcm"}          # never redact
DPSS_TOKEN = "<AGENCY>"                  # force redaction token for DPSS
PRONOUNS = {
    "he", "him", "his",
    "she", "her", "hers",
    "they", "them", "their", "theirs",
    "ze", "zir", "zirself", "xe", "xem", "xyr"
}
REDACT_FIRST_MIDDLE_INITIAL_NO_LAST = True  # catch "Jay S." style

# -----------------------------
# Utilities
# -----------------------------
def clean_cell(cell):
    if isinstance(cell, str):
        return ' '.join(cell.split())
    return cell

# -----------------------------
# PASS 1: Names-first redaction
# -----------------------------
def build_row_name_patterns(row):
    """
    Build robust regex patterns from one CSV row:
      - Redact any mention combo of First/Middle/Last (incl. initials & pairs).
      - Do NOT redact a single-letter middle by itself.
      - Catch possessives like "Doe's".
    """
    pats = []

    first  = str(row.get("First Name", "") or "").strip()
    middle = str(row.get("Middle Name", "") or "").strip()
    last   = str(row.get("Last Name", "") or "").strip()

    other_cols = [c for c in row.index if c.lower().startswith("other name")]
    others = [str(row.get(c, "") or "").strip() for c in other_cols if str(row.get(c, "") or "").strip()]

    def add_whole_word(name, tag="<PERSON>"):
        if name and len(name) > 1:
            rx = re.compile(rf"(?i)\b{re.escape(name)}\b(?:[’']s)?")
            pats.append((rx, tag))

    def add_pair_possessive(a, b, sep=r"\s+"):
        rx = re.compile(rf"(?i)\b{re.escape(a)}{sep}{re.escape(b)}\b(?:[’']s)?")
        pats.append((rx, "<PERSON>"))

    # Standalone names (multi-char only)
    add_whole_word(first)
    add_whole_word(last)
    if middle and len(middle) > 1:
        add_whole_word(middle)

    # Aliases in "Other Name i"
    for on in others:
        add_whole_word(on)

    if first and last:
        first_initial = first[0]
        # First Last (+ possessive)
        add_pair_possessive(first, last)

        if middle:
            if len(middle) == 1:
                # First M Last / First M. Last (+ possessive)
                rx = re.compile(rf"(?i)\b{re.escape(first)}\s+{re.escape(middle)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
                # F. M. Last (+ possessive)
                rx = re.compile(rf"(?i)\b{re.escape(first_initial)}\.?\s+{re.escape(middle)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
                # F. Last (+ possessive)
                rx = re.compile(rf"(?i)\b{re.escape(first_initial)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
            else:
                # First Middle Last (+ possessive)
                rx = re.compile(rf"(?i)\b{re.escape(first)}\s+{re.escape(middle)}\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
                # First M. Last / F. M. Last / F. Last
                m_initial = middle[0]
                rx = re.compile(rf"(?i)\b{re.escape(first)}\s+{re.escape(m_initial)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
                rx = re.compile(rf"(?i)\b{re.escape(first_initial)}\.?\s+{re.escape(m_initial)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
                rx = re.compile(rf"(?i)\b{re.escape(first_initial)}\.?\s+{re.escape(last)}\b(?:[’']s)?")
                pats.append((rx, "<PERSON>"))
    # "First M." without last (e.g., Jay S.)
    if REDACT_FIRST_MIDDLE_INITIAL_NO_LAST and first and middle and len(middle) == 1:
        rx = re.compile(rf"(?i)\b{re.escape(first)}\s+{re.escape(middle)}\.?(?=\b|[^A-Za-z])")
        pats.append((rx, "<PERSON>"))
        # Optionally "F. M."
        rx = re.compile(rf"(?i)\b{re.escape(first[0])}\.?\s*{re.escape(middle)}\.?(?=\b|[^A-Za-z])")
        pats.append((rx, "<PERSON>"))

    if last and first:
        # Initial + Last, even if first not spelled out elsewhere
        rx = re.compile(rf"(?i)\b{re.escape(first[0])}\.?\s+{re.escape(last)}\b(?:[’']s)?")
        pats.append((rx, "<PERSON>"))

    return pats

def redact_names_pass(text, all_row_patterns):
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pats in all_row_patterns:
        for rx, repl in pats:
            out = rx.sub(repl, out)
    return out

# -----------------------------
# PASS 2: PII via Presidio
# -----------------------------
def configure_nlp_engine(language='en'):
    configuration = {"nlp_engine_name": "spacy", "models": [{"lang_code": language, "model_name": "en_core_web_lg"}]}
    provider = NlpEngineProvider(nlp_configuration=configuration)
    return provider.create_engine()

def analyze_with_presidio(text, nlp_engine, language='en'):
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])

    # Force-recognize DPSS as ORGANIZATION (we'll remap output token below)
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="ORGANIZATION",
            name="force_dpss_org",
            patterns=[Pattern(name="DPSS", regex=r"\bDPSS\b", score=0.9)]
        )
    )

    results = analyzer.analyze(text=text, language=language)

    # Filter allowlist (YP/HCM) from results, regardless of detected type
    filtered = []
    for r in results:
        span = text[r.start:r.end]
        if span.strip().lower() in ALLOWLIST_TERMS:
            continue
        filtered.append(r)

    return filtered

def anonymize_text(text, analyzer_results, redact_pronouns=False):
    anonymizer = AnonymizerEngine()
    operators = {
        "PERSON":        OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "LOCATION":      OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        "ORGANIZATION":  OperatorConfig("replace", {"new_value": "<ORG>"}),  # DPSS normalized below
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
        "CREDIT_CARD":   OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        "DATE":          OperatorConfig("replace", {"new_value": r"\g<0>"}),  # no-op (avoid redacting "afternoon")
        "DATE_TIME":     OperatorConfig("replace", {"new_value": r"\g<0>"}),  # no-op
        "NUMBER":        OperatorConfig("replace", {"new_value": "<NUMBER>"}),
    }

    out = anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operators).text

    # Ensure DPSS becomes <AGENCY>, not <ORG>
    out = re.sub(r"\bDPSS\b", DPSS_TOKEN, out, flags=re.IGNORECASE)

    if redact_pronouns:
        pron_pat = r"\b(" + "|".join(sorted(PRONOUNS, key=len, reverse=True)) + r")\b"
        out = re.sub(pron_pat, "<PRONOUN>", out, flags=re.IGNORECASE)

    return out

# -----------------------------
# MAIN (with progress bars)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Anonymize text with names-first pass + Presidio, with progress bars.")
    parser.add_argument('--file', type=str, required=True, help='Path to input .csv with notes')
    parser.add_argument('--column', type=str, default='Notes', help='Column in CSV with the notes')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--provided_names', type=str, required=True, help='CSV with name columns: First Name, Middle Name, Last Name, Other Name 1..5')
    parser.add_argument('--redact_pronouns', action='store_true', help='If set, also redact pronouns')

    args = parser.parse_args()

    df = pd.read_csv(args.file)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in {args.file}")

    df[args.column] = df[args.column].apply(clean_cell)

    names_df = pd.read_csv(args.provided_names)
    expected_cols = [c for c in ["First Name", "Middle Name", "Last Name",
                                 "Other Name 1", "Other Name 2", "Other Name 3", "Other Name 4", "Other Name 5"]
                     if c in names_df.columns]
    if not expected_cols:
        raise ValueError("No expected name columns found in provided_names CSV.")

    # Build patterns per row (progress)
    all_row_patterns = []
    t = tqdm(total=len(names_df), desc="Building name patterns", unit="row") if TQDM_AVAILABLE else None
    for _, row in names_df[expected_cols].iterrows():
        all_row_patterns.append(build_row_name_patterns(row))
        if t: t.update(1)
    if t: t.close()

    # PASS 1: Names-first (progress)
    out_col_pass1 = f'{args.column}__pass1'
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="Pass 1: redacting names")
        df[out_col_pass1] = df[args.column].progress_apply(lambda t: redact_names_pass(t, all_row_patterns))
    else:
        df[out_col_pass1] = df[args.column].apply(lambda t: redact_names_pass(t, all_row_patterns))

    # PASS 2: PII via Presidio (progress)
    nlp_engine = configure_nlp_engine()
    def do_pass2(text):
        analyzed = analyze_with_presidio(text, nlp_engine)
        return anonymize_text(text, analyzed, redact_pronouns=args.redact_pronouns)

    out_col_final = f'{args.column}_anonymized'
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="Pass 2: PII anonymization")
        df[out_col_final] = df[out_col_pass1].progress_apply(do_pass2)
    else:
        df[out_col_final] = df[out_col_pass1].apply(do_pass2)

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__ == '__main__':
    main()
