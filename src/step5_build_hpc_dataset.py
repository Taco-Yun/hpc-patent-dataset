"""
Step 5 — Flatten the parsed records from step 4 into the final CSV dataset.

Reads  <data-dir>/interim/parsed_data.json        (JSON Lines from step 4)
Writes <data-dir>/processed/HPC_15to25_v2.csv     one row per patent (column list: see README)
Records that raise an error while being flattened (e.g. no classifications) are reported and skipped.
"""
import argparse
import os
import pandas as pd
import json
import re

from utils import DEFAULT_DATA_DIR, display_path


def extract_final_classifications(classifications) -> list:

    # The last entry of every classification group (skipping empty groups)
    results = [
        {
            'ipc': row[-1].get('code').split('/')[0],
            'ipc_description': row[-1].get('description')
        }
        for row in classifications if isinstance(row, list) and row
    ]

    return results
def extract_event_info(events):
    result = {
        'assignee': 'NA',
        'application_year':'NA',
        'application_date':"NA",
        'priority_date': 'NA',
        'priority_title': 'NA',
        'reassignment_assignee': 'NA',
        'reassignment_date': 'NA',
        'expiration_date': 'NA',
        'expiration_title': 'NA',
        'granted_title': 'NA',
        'granted_date': 'NA'
    }

    for event in events:
        event_type = event.get("type", "").lower()
        title = event.get("title", "").lower()

        # 1. filed + application
        if event_type == "filed" and "application" in title and result['assignee'] == 'NA':
            result['application_date'] = event.get('date',"NA")
            result['application_year'] = result['application_date'].split('-')[0]
            result['assignee'] = event.get("assigneeSearch", "NA")
            # print(title,result['application_date'], result['application_year'],)
            # exit()
            

        # 2. priority
        elif event_type == "priority" and result['priority_date'] == 'NA':
            result['priority_date'] = event.get("date", "NA")
            result['priority_title'] = event.get("title", "NA")

        # 3. reassignment
        elif event_type == "reassignment" and result['reassignment_assignee'] == 'NA':
            result['reassignment_assignee'] = event.get("assigneeSearch", "NA")
            result['reassignment_date'] = event.get("date", "NA")

        # 4. legal-status: anticipated/adjusted expiration
        elif event_type == "legal-status" and result['expiration_date'] == 'NA':
            if title in ["anticipated expiration", "adjusted expiration"]:
                result['expiration_date'] = event.get("date", "NA")
                result['expiration_title'] = event.get("title", "NA")

        # 5. granted
        elif event_type == "granted" and result['granted_date'] == 'NA':
            result['granted_date'] = event.get("date", "NA")
            result['granted_title'] = event.get("title", "NA")

        # ✅ Stop early once every field has been filled
        if all(value != 'NA' for value in result.values()):
            break

    return result


def process_claims(claims: list) -> dict:
    """
    Process the patent claims:
    1. Count the claims
    2. Keep the leading claim numbers
    3. Remove the line breaks inside each claim
    4. Join all claims into one string with the full-width semicolon "；" (U+FF1B), no line breaks

    Returns:
        dict: {
            "claim_count": int,
            "claims_text": str
        }
    """
    cleaned_claims = []
    for claim in claims:
        cleaned = claim.replace('\n', ' ').strip()
        cleaned_claims.append(cleaned)

    return {
        "claim_count": len(claims),
        "claims_text": "；".join(cleaned_claims)
    }

def extract_patent_citations_info(citations: list) -> dict:
    """
    Process patent_citations and return:
    1. The number of citations
    2. A list with the reduced fields publication_number, publication_date, assignee, title
    """
    result = []
    for item in citations:
        result.append({
            "publication_number": item.get("publication_number", "NA"),
            "publication_date": item.get("publication_date", "NA"),
            "assignee": item.get("assignee", "NA"),
            "title": item.get("title", "NA")
        })
    
    return {
        "citation_count": len(result),
        "citations": result
    }

def extract_ipc_code(classifications):
    if not classifications or not classifications[0]:
        return {'main_ipc_4': 'NA', 'main_ipc_description_4': 'NA'}

    for entry in classifications[0]:
        raw_code = entry.get('code', '')
        description = entry.get('description', '').strip() or 'NA'

        # ✅ Take the substring from the start up to the last capital letter (it must contain at least two capital letters)
        match = re.match(r'^([A-Z][A-Z0-9]*[A-Z])', raw_code)

        if match:
            code_part = match.group(1)
            if len(re.findall(r'[A-Z]', code_part)) >= 2:
                return {'main_ipc_4': code_part, 'main_ipc_description_4': description}

    return {'main_ipc_4': 'NA', 'main_ipc_description_4': 'NA'}





def load_and_parse_jsonl(filepath):
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Add your own processing logic here, for example:
                parsed = {
                    'patent_id': data.get('patent_id'),
                    'title': data.get('title', ''),
                    'abstract': data.get('abstract', ''),
                    'main_ipc': data.get('classifications', [])[0][-1].get('code', '').split('/')[0],
                    'main_ipc_description': data.get('classifications', [])[0][-1].get('description', ''),
                    **extract_ipc_code(data.get('classifications', [])),
                    'ipc_description': extract_final_classifications(data.get('classifications')),
                    'current_assignee': (data.get('current_assignee') or ['NA'])[0],
                    **extract_event_info(data.get('events')),
                    **data.get('description'),
                    **process_claims(data.get('claims')),
                    **extract_patent_citations_info(data.get('patent_citations')),
                    'non_patent_citation': len(data.get('non_patent_citations','')),
                    'applications_claiming_priority': data.get('applications_claiming_priority'),
                    'legal_events': data.get('legal_events'),
                    'patent_url': 'https://patents.google.com/patent/'+ data.get('patent_id')
                    
                }
                # print(parsed['patent_id'])
                # break
                records.append(parsed)
            except json.JSONDecodeError as e:
                # print(data)
                print(f"JSON parsing error: {e}")
            except Exception as e:
                print(f"Error, record skipped: {data.get('patent_id')} -> {e}")
                # exit()
    
    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory (default: <repo>/data).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = os.path.join(args.data_dir, "interim", "parsed_data.json")
    output_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "HPC_15to25_v2.csv")

    df = load_and_parse_jsonl(input_file)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {display_path(output_file)}")