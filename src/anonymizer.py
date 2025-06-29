import spacy
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import time
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
import pandas as pd
import os
import argparse
import re 
from spacy.pipeline import EntityRuler

# get a list of names -- full names and names of people, nicknames, 
# christian will give us a list of names

def enhance_spacy_with_rules(nlp):
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "PERSON", "pattern": [{"LOWER": "dr"}, {"IS_TITLE": True}]},
        {"label": "PERSON", "pattern": [{"LOWER": "mr"}, {"IS_TITLE": True}]},
    ]
    ruler.add_patterns(patterns)
    return nlp


def configure_nlp_engine(language='en'):
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return nlp_engine

def analyze_text_with_presidio(text, nlp_engine, language='en'):
    # titles_recognizer = PatternRecognizer(supported_entity="TITLE",
    #                                       deny_list=["Mr.","Mrs.","Miss"])
    # pronoun_recognizer = PatternRecognizer(supported_entity="PRONOUN",
    #                                       deny_list=["he", "He", "his", "His", "she", "She", "her" "hers", "Hers"])

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
    # analyzer.registry.add_recognizer(titles_recognizer)
    # analyzer.registry.add_recognizer(pronoun_recognizer)

    results = analyzer.analyze(text=text, language=language)
    return results

def analyze_text_with_spacy(text, nlp):
    doc = nlp(text)
    
    # Map spaCy labels to standardized Presidio-compatible labels
    entity_mapping = {
        "PERSON": "PERSON",
        # "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "CARDINAL": "NUMBER",
        "LOC": "LOCATION",
        "FAC": "LOCATION"
    }

    results = []
    for ent in doc.ents:
        if ent.label_ in entity_mapping:
            mapped_entity = entity_mapping[ent.label_]
            results.append(
                RecognizerResult(
                    entity_type=mapped_entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.85
                )
            )
    return results


def merge_results(presidio_results, spacy_results):
    presidio_results = presidio_results or []
    spacy_results = spacy_results or []

    # Start with spaCy results — these take priority
    unique = {(r.start, r.end): r for r in spacy_results}

    # Add Presidio results only if their span wasn't already seen
    for r in presidio_results:
        key = (r.start, r.end)
        if key not in unique:
            unique[key] = r

    return list(unique.values())

def anonymize_text(text, analyzer_results):
    anonymizer = AnonymizerEngine()
    operator_config = {
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        "DATE": OperatorConfig("replace", {"new_value": "<DATE_TIME>"}),
        "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE_TIME>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"})
    }
    return anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operator_config).text

# Define a cleaning function
def clean_cell(cell):
    if isinstance(cell, str):
        return ' '.join(cell.split()).lower()  # Removes excessive whitespace and newlines
    return cell

def redact_names(text, names_to_redact, replacement="[REDACTED]"):
    if pd.isna(text):
        return text

    for name in names_to_redact:
        # Skip empty strings
        if not name.strip():
            continue
        # Regex to match name as whole word
        pattern = r'\b{}\b'.format(re.escape(name))
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def main():
    parser = argparse.ArgumentParser(description="Anonymize text using spaCy + Presidio.")
    parser.add_argument('--file', type=str, default='notes.csv', help='Path to input .csv file with the original notes')
    parser.add_argument('--column', type=str, default='Notes', help='The name of the column in file which contains the notes')
    parser.add_argument('--output', type=str, default='anonymized.csv', help='Output CSV file path (required for CSV input)')
    parser.add_argument('--provided_names', type=str, default='names.csv', help='Path to the file which contains all the provided names, expecting columns: First Name, Middle Name, Last Name, Other Name 1, Other Name 2, Other Name 3, Other Name 4, Other Name 5')


    args = parser.parse_args()
    input_path = args.file
    output_path = args.output
    names_path = args.provided_names
    columns_with_names = ['First Name', 'Middle Name', 'Last Name', 'Other Name 1', 'Other Name 2', 'Other Name 3', 'Other Name 4', 'Other Name 5']

    # Setup NLP engines
    nlp_engine = configure_nlp_engine()
    spacy_nlp = spacy.load("en_core_web_lg")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in CSV.")
        
        df[args.column] = df[args.column].apply(clean_cell)


        # Anonymize each row in the specified column
        df[f'{args.column}_anonymized'] = df[args.column].apply(lambda text: anonymize_text(
            text,
            merge_results(
                analyze_text_with_presidio(text, nlp_engine),
                analyze_text_with_spacy(text, spacy_nlp)
            )
        ))

        # Anonymize in second path
        names_df = pd.read_csv(names_path)
        all_names_series = pd.concat([names_df[col] for col in columns_with_names if col in names_df.columns])
        # Clean names
        name_list = (
            all_names_series
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        print(f"Loaded {len(name_list)} unique names to check.")

        df[f'{args.column}_final_anonymized'] = df[f'{args.column}_anonymized'].apply(
            lambda x: redact_names(x, name_list)
)


 


        df.to_csv(output_path, index=False)
        print(f"Anonymized CSV saved to: {output_path}")

    else:
        raise ValueError("Unsupported file type. Only .csv are supported.")


if __name__ == '__main__':
    main()