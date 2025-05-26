# InfiniGram Query Generator & LLM Q&A Evaluator

This script processes a CSV file containing questions and answers. It performs two main tasks:
1.  **InfiniGram Analysis**: Extracts keywords (using a "rarest term" heuristic from the question and entities from the answer), constructs Conjunctive Normal Form (CNF) "AND" queries, and queries an InfiniGram API endpoint to get co-occurrence counts for these keywords.
2.  **LLM Evaluation**: Uses an OpenAI model to answer the questions and then self-evaluate its generated answer against the provided correct answer, outputting a correctness score (0 or 1).

The goal is to analyze relationships between corpus evidence (via InfiniGram) and an LLM's ability to answer questions correctly.

## Features

- Reads question-answer pairs from a CSV file.
- **LLM-based Q&A and Self-Correction**:
    - Utilizes an OpenAI model (e.g., GPT-3.5-turbo, GPT-4) to generate answers to questions.
    - Prompts the LLM to self-evaluate its answer against the ground truth answer, yielding a 0 (incorrect) or 1 (correct) score.
    - Caches LLM responses (`llm_eval_cache.json`) to avoid re-processing and save API costs.
- **InfiniGram Keyword Co-occurrence Analysis**:
    - Uses spaCy for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
    - Employs a heuristic to find the "rarest" significant term (entity or proper noun) in the question by checking individual term frequencies against an InfiniGram index (`query_type: 'count'`).
    - Extracts entities from the answer.
    - Combines the rarest question term with answer entities to form a specific "AND" query for InfiniGram (`query_type: 'find_cnf'`).
- Saves original data, LLM evaluation results, extracted InfiniGram query components, and InfiniGram API results to an output CSV file.

## Prerequisites

1.  **Python 3.7+**
2.  **InfiniGram API Endpoint**: Access to a running InfiniGram API endpoint.
3.  **OpenAI API Key**: An API key for OpenAI services.

## Setup

1.  **Clone the repository or download the script.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install `openai`, `requests`, `spacy`, `pandas`, `nltk`, `numpy`, `matplotlib`, and `seaborn`)*

4.  **Download the spaCy English model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set your OpenAI API Key:**
    The script expects the `OPENAI_API_KEY` as an environment variable.
    ```bash
    export OPENAI_API_KEY='your_actual_api_key_here'
    # On Windows (Command Prompt), use: set OPENAI_API_KEY=your_actual_api_key_here
    # Or (PowerShell): $env:OPENAI_API_KEY="your_actual_api_key_here"
    ```

## Configuration

Open `infinigram_query_generator.py` and review/configure the following variables within the `main()` function:

**InfiniGram Configuration:**
-   `INFINIGRAM_API_URL` (string): URL of your InfiniGram API query endpoint.
-   `INFINIGRAM_CNF_API_INDEX_NAME` (string): InfiniGram index for the main `find_cnf` queries (e.g., `"v4_olmo-2-0325-32b-instruct_llama"`).
-   `INFINIGRAM_SINGLE_TERM_COUNT_INDEX_NAME` (string): InfiniGram index for `query_type: 'count'` used by the rarity heuristic (can be the same as `INFINIGRAM_CNF_API_INDEX_NAME`).

**OpenAI LLM Configuration:**
-   `LLM_MODEL_NAME` (string): The OpenAI model to use for Q&A and self-correction (e.g., `"gpt-3.5-turbo"`, `"gpt-4.1-mini-2025-04-14"`).
-   `LLM_CACHE_FILE` (string): Path for the JSON file to cache LLM API responses (default: `"llm_eval_cache.json"`).

**Script Behavior Configuration:**
-   `csv_file_path` (string): Path to your input CSV file (default: `'simple_qa_test_set.csv'`). Must have `'problem'` and `'answer'` columns.
-   `output_csv_path` (string): Path where the results CSV will be saved (default: `'infinigram_query_results_heuristic_AND_llm.csv'`).
-   `max_rows_to_process` (integer or `None`): Max rows to process from input CSV. `None` for all.
-   `api_timeout_seconds` (integer): Timeout for InfiniGram API requests.

## Input CSV Format

The input CSV file must contain at least `problem` and `answer` columns.

## Running the Main Script (`infinigram_query_generator.py`)

Once configured, run the main data generation script from your terminal:

```bash
python infinigram_query_generator.py
```

The script will:
1.  Initialize OpenAI client (if API key is set) and load the LLM response cache.
2.  Initialize InfiniGram API session and spaCy model (if available).
3.  Process each row from the input CSV:
    -   **LLM Evaluation:**
        -   Generate an answer to the `problem` using the configured LLM (uses cache if available).
        -   Prompt the LLM to self-evaluate its answer against the `human_answer`, resulting in a 0/1 score (uses cache if available).
    -   **InfiniGram Analysis (if spaCy is available):**
        -   Identify candidate terms from the `problem` (entities or proper nouns).
        -   Query InfiniGram for individual frequencies of these terms to find the "rarest" one.
        -   Extract keywords (entities) from the `answer`.
        -   Construct an "AND" CNF query combining the rarest problem term and answer keywords.
        -   Send the CNF query to the InfiniGram API.
    -   Store all results for the row.
4.  Save the updated LLM response cache.
5.  Print a summary of processed results.
6.  Save detailed results to the output CSV.

## Output CSV Format

The output CSV file (default: `infinigram_query_results_heuristic_AND_llm.csv`) will contain the original data plus the following columns (order may vary slightly based on processing):

-   `problem`: Original problem text.
-   `answer`: Original human-provided answer text.
-   `llm_generated_answer`: The answer generated by the LLM.
-   `llm_correctness_score`: 0 (incorrect) or 1 (correct) based on LLM self-evaluation; -1 if skipped or error.
-   `rarest_problem_term`: The term from the problem identified as the rarest by the InfiniGram count heuristic.
-   `final_query_keywords`: Comma-separated string of unique keywords used in the final InfiniGram CNF query (rarest problem term + answer keywords).
-   `cnf_query_str`: The actual "AND" query string sent to InfiniGram for `find_cnf`.
-   `match_count`: The co-occurrence count (`cnt`) returned by InfiniGram for the `find_cnf` query.
-   `approx`: Boolean indicating if the InfiniGram `find_cnf` count is approximate.
-   `notes`: Any notes regarding the processing, such as API errors (InfiniGram or LLM), timeouts, or skipped steps.

## Visualizing Results (`visualize_results.py`)

A separate script, `visualize_results.py`, is provided to generate plots from the output CSV of `infinigram_query_generator.py`.
It creates side-by-side violin plots (with overlaid strip plots) to show the distribution of InfiniGram `match_count` values, 
separated for questions where the LLM self-correction score was 1 (correct - green) versus 0 (incorrect - red).
This helps visualize the relationship between InfiniGram evidence and LLM correctness.

**Running the Visualization Script:**

```bash
python visualize_results.py [path_to_your_output_csv_file] [--log]
```
- If `[path_to_your_output_csv_file]` is omitted, it defaults to `infinigram_query_results_heuristic_AND_llm.csv`.
- Use the optional `--log` flag to apply a logarithmic scale to the `match_count` axis, which can be useful if counts vary widely.
- The chart will be saved as `match_count_distribution.png` (or `match_count_distribution_log.png` if `--log` is used) in the current directory.

## Customizing Keyword Extraction and Heuristics

-   The `extract_keywords` function (for answer processing) and the logic for selecting `problem_terms_for_rarity` (in `main` of `infinigram_query_generator.py`) can be further customized (e.g., different spaCy entity types, POS tags, or other NLP techniques).
-   The LLM prompts in `main` for generation and self-correction can be modified to experiment with different instructional approaches. 