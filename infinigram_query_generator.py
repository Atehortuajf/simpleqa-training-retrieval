import csv
import json # For API request payload

# Try to import optional libraries for actual use
try:
    import requests
except ImportError:
    print("Warning: 'requests' library not found. Needed for API mode. Install with 'pip install requests'.")
    requests = None

try:
    import spacy
    NLP_SPACY = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except ImportError:
    print("Warning: 'spacy' library not found. Keyword extraction will use a basic placeholder. Install with 'pip install spacy'.")
    spacy = None
    NLP_SPACY = None
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Keyword extraction will use a basic placeholder.")
    print("Download it with: python -m spacy download en_core_web_sm")
    NLP_SPACY = None

# --- USER-DEFINED KEYWORD EXTRACTION --- (Now uses spaCy if available)
def extract_keywords(text: str) -> list[str]:
    """
    Extracts named entities from the text using spaCy if available.
    Otherwise, falls back to a basic placeholder keyword extraction.
    This function should take a string and return a list of keyword strings (phrases).
    """
    if NLP_SPACY:
        print(f"[spaCy NER] Processing text: '{text[:70]}...'")
        doc = NLP_SPACY(text)
        keywords = [ent.text for ent in doc.ents]
        print(f"[spaCy NER] Extracted entities: {keywords}")
        if not keywords:
            print("[spaCy NER] No entities found by spaCy. Consider refining NER or using fallback.")
        return keywords
    else:
        print(f"[Placeholder NER] Processing text: '{text[:70]}...'")
        words = text.replace('?', '').replace('.', '').replace(',', '').split()
        stopwords = {'the', 'a', 'is', 'in', 'of', 'to', 'and', 'who', 'what', 'was', 'were', 'how', 'did', 'according', 'for', 'on', 'at', 'by', 'with', 'an'}
        extracted = [word for word in words if len(word) > 3 and not word.isdigit() and word.lower() not in stopwords]
        
        if not extracted and len(words) > 0:
            extracted = [word for word in words if word.lower() not in stopwords]

        print(f"[Placeholder NER] Extracted: {extracted}")
        return extracted

# --- MAIN SCRIPT LOGIC ---
def main():
    # --- USER SETUP REQUIRED ---
    # 1. (Handled by spaCy in extract_keywords now, but you can customize it further)
    # 2. Install necessary libraries: pip install requests spacy
    # 3. Download spaCy model: python -m spacy download en_core_web_sm
    # 4. Ensure your InfiniGram API endpoint is running and accessible.
    # 5. Configure the API URL and index name below.

    # --- Configuration (User must set these for actual InfiniGram API use) ---
    INFINIGRAM_API_URL = "https://api.infini-gram.io/" # Replace with your InfiniGram API endpoint URL
    INFINIGRAM_API_INDEX_NAME = "v4_pileval_llama"    # Replace with the index name hosted by your API
    
    csv_file_path = 'simple_qa_test_set.csv'
    output_csv_path = 'infinigram_query_results_api.csv'
    max_rows_to_process = 25 # Set to None to process all rows
    api_timeout_seconds = 30 # Timeout for API requests

    if requests is None:
        print("ERROR: 'requests' library is not available. Cannot run in API mode.")
        print("Please install it: pip install requests")
        return
    if NLP_SPACY is None:
        print("ERROR: spaCy model is not available. Keyword extraction will be basic. API mode requires spaCy.")
        print("Ensure spaCy is installed and 'en_core_web_sm' model is downloaded.")

    print("--- USING InfiniGram API ---")
    api_session = requests.Session()

    # --- CSV Processing ---
    processed_results = []
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if max_rows_to_process is not None and i >= max_rows_to_process:
                    print(f"Processed {max_rows_to_process} rows. Halting as per max_rows_to_process limit.")
                    break

                problem = row.get('problem', '')
                answer = row.get('answer', '')

                print(f"\nProcessing row {i+1}:")
                print(f"  Problem: {problem}")
                print(f"  Answer: {answer}")

                problem_kws = extract_keywords(problem)
                answer_kws = extract_keywords(answer)
                
                all_keywords_list = sorted(list(set(problem_kws + answer_kws)))

                if not all_keywords_list:
                    print("  No keywords extracted. Skipping InfiniGram query.")
                    processed_results.append({
                        'problem': problem, 'answer': answer, 'keywords': [],
                        'cnf_query_str': 'N/A', 'match_count': 0, 'approx': None,
                        'notes': 'No keywords extracted'
                    })
                    continue
                
                print(f"  Combined unique keywords: {all_keywords_list}")

                # Construct the payload for the API
                query_string = " OR ".join(all_keywords_list)  # Join keywords with ' OR '
                payload = {
                    "index": INFINIGRAM_API_INDEX_NAME,
                    "query_type": "find_cnf",
                    "query": query_string
                }

                try:
                    response = api_session.post(INFINIGRAM_API_URL, json=payload, timeout=api_timeout_seconds)
                    response.raise_for_status()
                    result_data = response.json()
                    
                    print(f"  InfiniGram API Result: {result_data}")
                    processed_results.append({
                        'problem': problem, 'answer': answer, 'keywords': ", ".join(all_keywords_list),
                        'cnf_query_str': query_string,
                        'match_count': result_data.get('cnt'),
                        'approx': result_data.get('approx'),
                        'notes': f"Latency: {result_data.get('latency', 'N/A')}"
                    })
                except requests.exceptions.Timeout:
                    print(f"  Error: API request timed out after {api_timeout_seconds} seconds.")
                    processed_results.append({
                        'problem': problem, 'answer': answer, 'keywords': ", ".join(all_keywords_list),
                        'cnf_query_str': query_string, 'match_count': None, 'approx': None,
                        'notes': 'API request timed out'
                    })
                except requests.exceptions.RequestException as e:
                    print(f"  Error during InfiniGram API query: {e}")
                    processed_results.append({
                        'problem': problem, 'answer': answer, 'keywords': ", ".join(all_keywords_list),
                        'cnf_query_str': query_string, 'match_count': None, 'approx': None,
                        'notes': f'InfiniGram API query error: {str(e)}'
                    })

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")
        return

    # --- Output Results ---
    print("\n\n--- Summary of Processed Results ---")
    for res_item in processed_results:
        print(f"  Problem: '{res_item['problem'][:50]}...' | Keywords: '{res_item['keywords']}' | Match Count: {res_item['match_count']} (Approx: {res_item['approx']}) {res_item['notes']}")

    if processed_results:
        try:
            with open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
                fieldnames = processed_results[0].keys()
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_results)
            print(f"\nResults saved to '{output_csv_path}'")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    else:
        print("\nNo results to save.")

if __name__ == '__main__':
    main() 