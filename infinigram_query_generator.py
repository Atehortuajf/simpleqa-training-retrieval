import csv
import json # For API request payload
import os
try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' library not found. LLM evaluation features will not work.")
    print("Install with: pip install openai")
    OpenAI = None

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

# --- LLM Cache Functions ---
def load_llm_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load LLM cache from {cache_path}: {e}")
    return {}

def save_llm_cache(cache_data: dict, cache_path: str):
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save LLM cache to {cache_path}: {e}")

# --- OpenAI API Interaction ---
def get_llm_response(client, prompt_text: str, model_name: str, 
                     llm_cache: dict, cache_key: str, 
                     is_self_correct_prompt: bool = False) -> str | None:
    if cache_key in llm_cache:
        # print(f"  [LLM Cache] Found in cache for key: {cache_key[:50]}...")
        return llm_cache[cache_key]

    if not client:
        print("  [LLM Error] OpenAI client not initialized.")
        return None
        
    # print(f"  [LLM API] Querying {model_name} for key: {cache_key[:50]}...")
    # print(f"  [LLM API] Prompt: {prompt_text[:100]}...")
    try:
        messages = [{"role": "user", "content": prompt_text}]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0, # Always use deterministic temperature for cache consistency
            max_tokens=5 if is_self_correct_prompt else 150    # Very short for 0/1, longer for answers
        )
        response_text = completion.choices[0].message.content.strip()
        
        if is_self_correct_prompt: # Expect '0' or '1'
            if response_text in ['0', '1']:
                llm_cache[cache_key] = response_text
                return response_text
            else:
                print(f"  [LLM Warning] Self-correction prompt did not return '0' or '1'. Got: '{response_text}'")
                return None # Or a specific error code like -1
        else:
            llm_cache[cache_key] = response_text
            return response_text
            
    except Exception as e:
        print(f"  [LLM Error] OpenAI API request failed: {e}")
        return None

# --- USER-DEFINED KEYWORD EXTRACTION --- (Uses spaCy if available)
def extract_keywords(text: str) -> list[str]:
    """
    Extracts named entities from the text using spaCy if available.
    Otherwise, falls back to a basic placeholder keyword extraction.
    """
    if NLP_SPACY:
        # print(f"[spaCy NER] Processing text for entities: '{text[:70]}...'")
        doc = NLP_SPACY(text)
        # Using entities directly is often better than just PROPN tokens for multi-word concepts
        keywords = [ent.text.strip() for ent in doc.ents if ent.text.strip()] 
        # print(f"[spaCy NER] Extracted entities: {keywords}")
        if not keywords:
            # Fallback to non-entity nouns and proper nouns if no entities found
            keywords = [token.text.strip() for token in doc if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop and not token.is_punct and len(token.text.strip()) > 1]
            # print(f"[spaCy NER] No entities found, using NOUN/PROPN tokens: {keywords}")
        return list(set(keywords)) # Return unique keywords
    else:
        # Fallback to basic placeholder if spaCy is not available
        print(f"[Placeholder NER] Processing text: '{text[:70]}...'")
        words = text.replace('?', '').replace('.', '').replace(',', '').split()
        stopwords = {'the', 'a', 'is', 'in', 'of', 'to', 'and', 'who', 'what', 'was', 'were', 'how', 'did', 'according', 'for', 'on', 'at', 'by', 'with', 'an'}
        extracted = [word for word in words if len(word) > 3 and not word.isdigit() and word.lower() not in stopwords]
        
        if not extracted and len(words) > 0:
            extracted = [word for word in words if word.lower() not in stopwords]
        # print(f"[Placeholder NER] Extracted: {extracted}")
        return list(set(extracted))

# --- InfiniGram Helper for Single Term Count ---
def get_single_term_frequency(term: str, api_session, api_url: str, index_name: str, timeout: int) -> int | None:
    """
    Queries InfiniGram for the frequency of a single term using query_type: 'count'.
    Returns the count as an integer, or None if an error occurs or count is not found.
    """
    if not term or not term.strip():
        return None
    
    payload = {
        "index": index_name,
        "query_type": "count",
        "query": term
    }
    try:
        # print(f"    [Term Freq] Querying count for: '{term}' on index '{index_name}'")
        response = api_session.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result_data = response.json()
        count = result_data.get('count')
        if isinstance(count, int):
            # print(f"    [Term Freq] Count for '{term}': {count}")
            return count
        else:
            # print(f"    [Term Freq] Unexpected count format for '{term}': {count}")
            return 0 # Treat as 0 if count is not an int (e.g. term not found in index)
    except requests.exceptions.Timeout:
        print(f"    [Term Freq] Timeout error getting count for '{term}'.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    [Term Freq] Request error getting count for '{term}': {e}")
        return None
    except json.JSONDecodeError:
        print(f"    [Term Freq] JSON decode error for response of '{term}'.")
        return None


# --- MAIN SCRIPT LOGIC ---
def main():
    # --- USER SETUP REQUIRED ---
    # (Customize API URL, index names, file paths, etc. below)

    # --- Configuration ---
    INFINIGRAM_API_URL = "https://api.infini-gram.io/"
    # Index for the main find_cnf query
    INFINIGRAM_CNF_API_INDEX_NAME = "v4_olmo-2-0325-32b-instruct_llama"
    # Index for single term frequency counts (heuristic). Default to same as CNF index if not specified.
    # The user's heuristic script used 'v4_rpj_llama_s4'. If different, set it here.
    INFINIGRAM_SINGLE_TERM_COUNT_INDEX_NAME = "v4_olmo-2-0325-32b-instruct_llama" # Or "v4_rpj_llama_s4" if preferred for counts

    csv_file_path = 'simple_qa_test_set.csv'
    output_csv_path = 'infinigram_query_results_heuristic_AND_llm.csv' # Changed output file name
    max_rows_to_process = 100 
    api_timeout_seconds = 30

    # --- OpenAI LLM Configuration ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_MODEL_NAME = "gpt-4.1-mini-2025-04-14"  # Or "gpt-4", "gpt-4-turbo" etc.
    LLM_CACHE_FILE = "llm_eval_cache.json"
    
    openai_client = None
    if OpenAI and OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"--- OpenAI client initialized. Model: {LLM_MODEL_NAME} ---")
        except Exception as e:
            print(f"--- Failed to initialize OpenAI client: {e} ---")
            openai_client = None 
    else:
        if not OpenAI:
            print("--- OpenAI library not loaded. LLM evaluation will be skipped. ---")
        if not OPENAI_API_KEY:
            print("--- OPENAI_API_KEY environment variable not set. LLM evaluation will be skipped. ---")
            print("--- Please set it: export OPENAI_API_KEY='your_key_here' ---")

    if requests is None:
        print("ERROR: 'requests' library is not available. Cannot run in API mode.")
        return
    if NLP_SPACY is None:
        print("ERROR: spaCy model is not available. Keyword extraction will be basic.")
        # This script now heavily relies on spaCy for the heuristic. Exit or warn strongly.
        return


    print("--- USING InfiniGram API with Rarest Proper Noun Heuristic (AND queries) & LLM Evaluation ---")
    api_session = requests.Session()
    llm_cache_data = load_llm_cache(LLM_CACHE_FILE)

    processed_results = []
    potential_fieldnames = [
        'problem', 'answer', 'llm_generated_answer', 'llm_correctness_score',
        'rarest_problem_term', 'answer_keywords', 'final_query_keywords', 
        'cnf_query_str', 'match_count', 'approx', 'notes', 'heuristic_cache_status'
    ]

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row_data in enumerate(reader):
                if max_rows_to_process is not None and i >= max_rows_to_process:
                    print(f"Processed {max_rows_to_process} rows. Halting.")
                    break

                problem = row_data.get('problem', '').strip()
                human_answer = row_data.get('answer', '').strip()
                current_row_results = {key: None for key in potential_fieldnames}
                current_row_results.update({
                    'problem': problem, 'answer': human_answer,
                    'llm_generated_answer': "N/A_SKIPPED", 'llm_correctness_score': -1,
                    'rarest_problem_term': "N/A", 'answer_keywords': "N/A",
                    'final_query_keywords': "N/A", 'cnf_query_str': "N/A", 'notes': "Skipped",
                    'heuristic_cache_status': 'Not applicable'
                })
                print(f"\nProcessing row {i+1}: {problem[:80]}...")
                
                # --- LLM Evaluation ---
                if openai_client:
                    # 1. Get LLM's answer to the problem
                    answer_prompt = f"Question: {problem}\nAnswer:"
                    answer_cache_key = f"ans_{LLM_MODEL_NAME}_{hash(problem)}"
                    llm_generated_answer = get_llm_response(openai_client, answer_prompt, LLM_MODEL_NAME, 
                                                            llm_cache_data, answer_cache_key)
                    current_row_results['llm_generated_answer'] = llm_generated_answer if llm_generated_answer else "N/A_FAILED"
                    if llm_generated_answer:
                        print(f"  [LLM] Generated Answer: {llm_generated_answer[:70]}...")
                        # 2. LLM self-corrects
                        correction_prompt = (
                            f"You are an evaluation assistant. Your task is to determine if an AI's answer to a question is correct.\n\n"
                            f"The original question was: \"{problem}\"\n\n"
                            f"The AI's proposed answer was: \"{llm_generated_answer}\"\n\n"
                            f"The correct human-provided answer is: \"{human_answer}\"\n\n"
                            f"Considering the correct human-provided answer, is the AI's proposed answer a correct and accurate response to the original question?\n"
                            f"Respond with only a single digit: '1' for yes (correct) or '0' for no (incorrect). Do not provide any other text or explanation."
                        )
                        correction_cache_key = f"corr_{LLM_MODEL_NAME}_{hash(problem + llm_generated_answer + human_answer)}"
                        score_str = get_llm_response(openai_client, correction_prompt, LLM_MODEL_NAME, 
                                                     llm_cache_data, correction_cache_key, is_self_correct_prompt=True)
                        if score_str in ['0', '1']:
                            current_row_results['llm_correctness_score'] = int(score_str)
                            print(f"  [LLM] Self-correction score: {current_row_results['llm_correctness_score']}")
                        else:
                            print("  [LLM] Self-correction did not yield 0 or 1.")
                            current_row_results['llm_correctness_score'] = -1 # Indicate error or invalid response
                    else:
                        print("  [LLM] Could not generate initial answer, skipping self-correction.")
                        current_row_results['llm_correctness_score'] = -1 
                else:
                    print("  [LLM] Skipping LLM evaluation (client not available).")
                    current_row_results['llm_generated_answer'] = "N/A_SKIPPED"
                    current_row_results['llm_correctness_score'] = -1 # Indicate skipped

                # --- Rarest Proper Noun Heuristic ---
                rarest_problem_term = None
                if NLP_SPACY:
                    problem_doc = NLP_SPACY(problem)
                    
                    # Define the set of relevant entity labels
                    relevant_entity_labels = {
                        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", 
                        "EVENT", "WORK_OF_ART", "FAC", "NORP", "LAW"
                    }
                    
                    # Extract entities from the problem text
                    problem_entities = [
                        ent.text.strip() 
                        for ent in problem_doc.ents 
                        if ent.text.strip() and ent.label_ in relevant_entity_labels
                    ]
                    
                    if not problem_entities: # Fallback to PROPN tokens if no relevant entities
                        problem_entities = sorted(list(set(
                            token.text.strip() 
                            for token in problem_doc 
                            if token.pos_ == 'PROPN' and 
                               not token.is_stop and 
                               not token.is_punct and 
                               len(token.text.strip()) > 1
                        )))

                    if problem_entities:
                        print(f"  [Heuristic] Candidate problem terms for rarity check: {problem_entities}")
                        term_counts = []
                        for p_term in problem_entities:
                            count = get_single_term_frequency(p_term, api_session, INFINIGRAM_API_URL, INFINIGRAM_SINGLE_TERM_COUNT_INDEX_NAME, api_timeout_seconds)
                            if count is not None: # Only consider terms for which count was successful
                                term_counts.append({'term': p_term, 'count': count})
                        
                        if term_counts:
                            term_counts.sort(key=lambda x: (x['count'], x['term'])) # Sort by count, then term
                            rarest_problem_term = term_counts[0]['term']
                            min_term_count = term_counts[0]['count']
                            print(f"  [Heuristic] Selected rarest term from problem: '{rarest_problem_term}' (Count: {min_term_count})")
                        else:
                            print("  [Heuristic] Could not get counts for any problem terms.")
                    else:
                        print("  [Heuristic] No relevant entities or PROPN tokens found in problem for rarity check.")
                
                # --- Keyword Extraction for Answer ---
                answer_kws = extract_keywords(human_answer)
                print(f"  Answer keywords: {answer_kws}")

                # --- Construct Query Components ---
                query_components = []
                if rarest_problem_term:
                    query_components.append(rarest_problem_term)
                else: # Fallback if no rarest term was found
                    print("  [Heuristic] No rarest term from problem, using general keyword extraction for problem.")
                    problem_kws_fallback = extract_keywords(problem)
                    query_components.extend(problem_kws_fallback)
                
                query_components.extend(answer_kws)
                
                # Filter out empty strings and ensure uniqueness
                all_keywords_list = sorted(set(qc for qc in query_components if qc and qc.strip()))

                if not all_keywords_list:
                    print("  No keywords for query. Skipping InfiniGram query.")
                    current_row_results['notes'] = 'No keywords for CNF query'
                    processed_results.append(current_row_results)
                    continue
                
                print(f"  Final keywords for query: {all_keywords_list}")
                # Using "AND" based on the heuristic's goal of finding specific co-occurrence
                query_string = " AND ".join(all_keywords_list)
                print(f"  Constructed CNF (AND) Query: \"{query_string}\"")

                # InfiniGram Heuristic with Caching
                rarest_term_found_this_row = None
                extracted_answer_kws_list = [] # Keep as list for processing
                all_keywords_list = []
                cnf_query = "N/A"
                ig_notes_for_row = "Skipped (spaCy not loaded)"
                heuristic_cache_status = "Skipped (spaCy not loaded)"

                if NLP_SPACY:
                    heuristic_cache_key = f"heuristic_v2_{hash(problem + human_answer)}"
                    loaded_heuristic_from_cache = False

                    if heuristic_cache_key in llm_cache_data:
                        cached_data = llm_cache_data[heuristic_cache_key]
                        rarest_term_found_this_row = cached_data.get('rarest_problem_term')
                        # Split stringified lists back into lists
                        extracted_answer_kws_list = cached_data.get('answer_keywords_str', "").split(',') if cached_data.get('answer_keywords_str') else []
                        all_keywords_list = cached_data.get('final_query_keywords_str', "").split(',') if cached_data.get('final_query_keywords_str') else []
                        cnf_query = cached_data.get('cnf_query_str', "N/A")
                        heuristic_cache_status = f"Loaded from cache (Key: {heuristic_cache_key[:15]}...)"
                        # print(f"  [Heuristic Cache] Loaded: {heuristic_cache_status}")
                        loaded_heuristic_from_cache = True
                    
                    if not loaded_heuristic_from_cache:
                        heuristic_cache_status = "Generated (Not in cache)"
                        # print(f"  [Heuristic Cache] {heuristic_cache_status}")
                        problem_doc = NLP_SPACY(problem)
                        relevant_entity_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "NORP", "LAW"}
                        problem_terms_for_rarity = [
                            ent.text.strip() for ent in problem_doc.ents 
                            if ent.text.strip() and ent.label_ in relevant_entity_labels
                        ]
                        if not problem_terms_for_rarity:
                            problem_terms_for_rarity = sorted(
                                set(
                                    token.text.strip() for token in problem_doc 
                                    if token.pos_ == 'PROPN' and not token.is_stop and not token.is_punct and len(token.text.strip()) > 1
                                )
                            )
                        
                        if problem_terms_for_rarity:
                            term_counts_for_rarity = []
                            for p_term in problem_terms_for_rarity:
                                count = get_single_term_frequency(p_term, api_session, INFINIGRAM_API_URL, INFINIGRAM_SINGLE_TERM_COUNT_INDEX_NAME, api_timeout_seconds)
                                if count is not None: term_counts_for_rarity.append({'term': p_term, 'count': count})
                            if term_counts_for_rarity:
                                term_counts_for_rarity.sort(key=lambda x: (x['count'], x['term']))
                                rarest_term_found_this_row = term_counts_for_rarity[0]['term']
                        
                        extracted_answer_kws_list = extract_keywords(human_answer)
                        
                        query_components = []
                        if rarest_term_found_this_row: query_components.append(rarest_term_found_this_row)
                        else: query_components.extend(extract_keywords(problem))
                        query_components.extend(extracted_answer_kws_list)
                        all_keywords_list = sorted(set(qc for qc in query_components if qc and qc.strip()))
                        
                        if all_keywords_list: cnf_query = " AND ".join(all_keywords_list)
                        else: cnf_query = "N/A"

                        # Store generated heuristic components in cache
                        llm_cache_data[heuristic_cache_key] = {
                            'rarest_problem_term': rarest_term_found_this_row,
                            'answer_keywords_str': ",".join(extracted_answer_kws_list),
                            'final_query_keywords_str': ",".join(all_keywords_list),
                            'cnf_query_str': cnf_query
                        }
                    
                    # Update current_row_results with heuristic data (cached or generated)
                    current_row_results['rarest_problem_term'] = rarest_term_found_this_row or "N/A"
                    current_row_results['answer_keywords'] = ",".join(extracted_answer_kws_list) if extracted_answer_kws_list else "N/A"
                    current_row_results['final_query_keywords'] = ",".join(all_keywords_list) if all_keywords_list else "N/A"
                    current_row_results['cnf_query_str'] = cnf_query
                    current_row_results['heuristic_cache_status'] = heuristic_cache_status

                    # Perform InfiniGram API call if a valid CNF query was formed
                    if cnf_query != "N/A" and all_keywords_list:
                        # print(f"  [IG Query]: \"{cnf_query}\"")
                        payload = {"index": INFINIGRAM_CNF_API_INDEX_NAME, "query_type": "find_cnf", "query": cnf_query}
                        try:
                            response = api_session.post(INFINIGRAM_API_URL, json=payload, timeout=api_timeout_seconds)
                            response.raise_for_status()
                            ig_result_data = response.json()
                            current_row_results['match_count'] = ig_result_data.get('cnt')
                            current_row_results['approx'] = ig_result_data.get('approx')
                            ig_notes_for_row = f"InfiniGram Latency: {ig_result_data.get('latency', 'N/A')}"
                        except requests.exceptions.Timeout:
                            ig_notes_for_row = 'InfiniGram API request timed out'
                        except requests.exceptions.RequestException as e:
                            ig_notes_for_row = f'InfiniGram API query error: {str(e)}'
                    elif not all_keywords_list:
                        ig_notes_for_row = 'No keywords for CNF query'
                        current_row_results['match_count'] = 0
                        current_row_results['approx'] = False
                
                current_row_results['notes'] = ig_notes_for_row # Set notes based on IG processing
                processed_results.append(current_row_results)

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")
        # Optionally save partial cache even on error
        if openai_client: save_llm_cache(llm_cache_data, LLM_CACHE_FILE)
        return
    finally:
        if openai_client or llm_cache_data: # Save if client was attempted or cache has data
            save_llm_cache(llm_cache_data, LLM_CACHE_FILE)
            print(f"--- LLM and Heuristic cache saved to {LLM_CACHE_FILE} ---")


    # Output Results
    if processed_results:
        header = potential_fieldnames
        print(f"\n--- Preparing to write CSV: {output_csv_path} ---")
        print(f"Number of rows to write: {len(processed_results)}")
        print(f"CSV Header to be used: {header}")
        if processed_results:
            # Check keys of the first row against the header
            first_row_keys = set(processed_results[0].keys())
            header_set = set(header)
            if first_row_keys != header_set:
                print(f"Warning: Keys in first row ({len(first_row_keys)} keys) do not perfectly match header ({len(header_set)} keys).")
                if len(header_set - first_row_keys) > 0: print(f"  Missing from first row (but in header): {sorted(list(header_set - first_row_keys))}")
                if len(first_row_keys - header_set) > 0: print(f"  Extra in first row (but not in header): {sorted(list(first_row_keys - header_set))}")

        print("\n\n--- Summary of Processed Results (Console) ---")
        for res_item in processed_results:
            print(f"  P: '{res_item.get('problem','.')[:30]}...' LLM OK?: {res_item.get('llm_correctness_score', 'N/A')}, IG Count: {res_item.get('match_count', 'N/A')}, Heuristic Cache: {res_item.get('heuristic_cache_status','N/A')[:15]}")

        try:
            with open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
                # extrasaction='ignore' means if a key is in a row_dict but not in fieldnames, it's skipped.
                # If a key is in fieldnames but not in a row_dict, it's written as an empty field (which is desired here).
                writer = csv.DictWriter(outfile, fieldnames=header, extrasaction='raise') # Changed to 'raise' to catch errors if keys are missing in rows but present in header
                writer.writeheader()
                writer.writerows(processed_results)
            print(f"\nResults saved to '{output_csv_path}'")
        except ValueError as ve:
            print(f"Error saving results to CSV (ValueError): {ve}")
            print("This usually means a row in processed_results is missing a key defined in the header, or has an extra key and extrasaction='raise'.")
            # Detailed row-by-row check if a ValueError occurs
            for i_debug, item_debug in enumerate(processed_results):
                missing_keys_in_row = [k_debug for k_debug in header if k_debug not in item_debug]
                extra_keys_in_row = [k_debug for k_debug in item_debug if k_debug not in header]
                if missing_keys_in_row or extra_keys_in_row:
                    if i_debug < 5 : print(f"Row {i_debug} Missing: {missing_keys_in_row} Extra: {extra_keys_in_row}")
                    else: print("Further row checks stopped."); break 
        except Exception as e: 
            print(f"Error saving results to CSV: {e}")
            print("Please check if all rows in memory contain all keys listed in the header.")
            if processed_results:
                for i, item_debug in enumerate(processed_results):
                    missing_keys_in_row = [k for k in header if k not in item_debug]
                    if missing_keys_in_row:
                        print(f"Row {i} is missing keys: {missing_keys_in_row}")
                        # print(f"Row {i} data: {item_debug}") # Uncomment for more detail
                        if i > 5 : break # Print details for a few rows only
    else:
        print("\nNo results to save.")

if __name__ == '__main__':
    # Add a reminder for OpenAI setup if not already handled
    if not OpenAI or not os.environ.get("OPENAI_API_KEY"):
        print("\n--- REMINDER FOR LLM EVALUATION ---")
        if not OpenAI:
            print("Please install the OpenAI library: pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            print("Please set your OpenAI API key as an environment variable:")
            print("  export OPENAI_API_KEY='your_actual_api_key'")
        print("LLM evaluation will be skipped if these are not set up.")
        print("-------------------------------------\n")
    main() 