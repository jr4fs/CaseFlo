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



def configure_nlp_engine(language='en'):
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return nlp_engine

def analyze_text_with_presidio(text, nlp_engine, language='en'):
    titles_recognizer = PatternRecognizer(supported_entity="TITLE",
                                          deny_list=["Mr.","Mrs.","Miss"])
    pronoun_recognizer = PatternRecognizer(supported_entity="PRONOUN",
                                          deny_list=["he", "He", "his", "His", "she", "She", "her" "hers", "Hers"])

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
    analyzer.registry.add_recognizer(titles_recognizer)
    analyzer.registry.add_recognizer(pronoun_recognizer)
    results = analyzer.analyze(text=text, language=language)
    return results


def anonymize_entities(text, model_name):
    nlp = spacy.load(model_name)
    # Process the text with spaCy NER
    doc = nlp(text)
    # Define a mapping for entity labels to their anonymized values
    entity_mapping = {
        "PERSON": "<PERSON>",
        # "ORG": "<ORGANIZATION>",
        "GPE": "<LOCATION>",
        "DATE": "<DATE_TIME>",
        "CARDINAL": "<NUMBER>"
    }
    # Create a list to store the anonymized text segments
    anonymized_text_segments = []
    # Track the end of the last entity to handle non-entity text
    last_end = 0
    for ent in doc.ents:
        # Append the text before the current entity
        anonymized_text_segments.append(text[last_end:ent.start_char])
        # Append the anonymized value for the current entity
        anonymized_value = entity_mapping.get(ent.label_, ent.text)
        anonymized_text_segments.append(anonymized_value)
        # Update the end position of the last entity
        last_end = ent.end_char
    # Append any remaining text after the last entity
    anonymized_text_segments.append(text[last_end:])
    # Combine all the text segments into the final anonymized text
    anonymized_text = "".join(anonymized_text_segments)
    return anonymized_text

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
        # "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ORGANIZATION>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"})
    }
    return anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operator_config).text


def main():
    parser = argparse.ArgumentParser(description="Anonymize text using spaCy + Presidio.")
    parser.add_argument('--file', type=str, required=True, help='Path to input .csv file')
    parser.add_argument('--column', type=str, default='Notes', help='Column to anonymize (only for CSV)')
    parser.add_argument('--output', type=str, default='anonymized.csv', help='Output CSV file path (required for CSV input)', required=True)

    args = parser.parse_args()
    input_path = args.file
    output_path = args.output

    # Setup NLP engines
    nlp_engine = configure_nlp_engine()
    spacy_nlp = spacy.load("en_core_web_lg")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in CSV.")

        # Anonymize each row in the specified column
        df[f'{args.column}_anonymized'] = df[args.column].apply(lambda text: anonymize_text(
            text,
            merge_results(
                analyze_text_with_presidio(text, nlp_engine),
                analyze_text_with_spacy(text, spacy_nlp)
            )
        ))
        df.to_csv(output_path, index=False)
        print(f"Anonymized CSV saved to: {output_path}")

    else:
        raise ValueError("Unsupported file type. Only .csv are supported.")


if __name__ == '__main__':
    main()