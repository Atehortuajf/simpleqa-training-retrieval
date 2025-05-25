# InfiniGram Query Generator for Q&A Analysis
**NOTE:** this repo is mega AI slop lol. 
This script processes a CSV file containing questions and answers, extracts keywords using spaCy, and then queries an InfiniGram API endpoint to count the co-occurrence of these keywords. The goal is to analyze the relationship between the presence of question/answer components in a corpus (via InfiniGram counts) and a model's ability to answer the question.

## Features

- Reads question-answer pairs from a CSV file.
- Uses spaCy for Named Entity Recognition (NER) to extract keywords from questions and answers.
- Falls back to a basic keyword extraction method if spaCy or its model is unavailable.
- Constructs Conjunctive Normal Form (CNF) queries based on the extracted keywords.
- Queries a configured InfiniGram API endpoint to get co-occurrence counts for these keywords.
- Saves the original data along with extracted keywords, InfiniGram query details, and results to an output CSV file.
- Includes a mock mode for testing the script's workflow without a live InfiniGram API or fully configured tokenizer/NER.

## Prerequisites

1.  **Python 3.7+**
2.  **InfiniGram API Endpoint**: You need access to a running InfiniGram API endpoint and the name of the index you wish to query.
3.  **Hugging Face Tokenizer**: A tokenizer compatible with the InfiniGram index you are querying (e.g., from the Hugging Face Hub).

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

4.  **Download the spaCy English model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
    *(The script uses `en_core_web_sm` by default. You can modify `infinigram_query_generator.py` to use other spaCy models if needed.)*

## Configuration

Open `infinigram_query_generator.py` and configure the following variables within the `main()` function:

-   `use_mock_infinigram_api` (boolean):
    -   Set to `False` to use the actual InfiniGram API and your configured tokenizer/spaCy setup.
    -   Set to `True` (default) to use mock objects for the API, tokenizer, and NER. This is useful for initial testing or if you don't have a live API yet.
-   `tokenizer_name_or_path` (string):
    -   The Hugging Face model name or local path for the tokenizer (e.g., `"meta-llama/Llama-2-7b-hf"`). This should match the tokenizer used for your InfiniGram index.
-   `INFINIGRAM_API_URL` (string):
    -   The full URL of your InfiniGram API query endpoint (e.g., `"http://localhost:8000/query"`).
-   `INFINIGRAM_API_INDEX_NAME` (string):
    -   The name of the InfiniGram index you want to query on the API (e.g., `"v4_pileval_llama"`).
-   `csv_file_path` (string):
    -   Path to your input CSV file (default: `'simple_qa_test_set.csv'`).
    -   The CSV should have at least `'problem'` and `'answer'` columns.
-   `output_csv_path` (string):
    -   Path where the results CSV will be saved (default: `'infinigram_query_results_api.csv'`).
-   `max_rows_to_process` (integer or `None`):
    -   Maximum number of rows to process from the input CSV. Set to `None` to process all rows (default: `5`).
-   `api_timeout_seconds` (integer):
    -   Timeout in seconds for requests to the InfiniGram API (default: `30`).

## Input CSV Format

The input CSV file (e.g., `simple_qa_test_set.csv`) must contain at least two columns:
-   `problem`: The question text.
-   `answer`: The answer text.

Example `simple_qa_test_set.csv` row:
```csv
metadata,problem,answer
"{'topic': 'Science and technology', ...}","Who received the IEEE Frank Rosenblatt Award in 2010?","Michio Sugeno"
```

## Running the Script

Once configured, run the script from your terminal:

```bash
python infinigram_query_generator.py
```

The script will:
1.  Print warnings if `transformers`, `requests`, or `spacy` (or its model) are not found (and fall back to mocks/placeholders if `use_mock_infinigram_api` is `True` or if the component is non-critical for mock mode).
2.  Load the tokenizer and spaCy model (if available and not in full mock mode).
3.  Process each row from the input CSV:
    -   Extract keywords from the `problem` and `answer` fields using spaCy NER (or fallback).
    -   Tokenize these keywords.
    -   Construct a CNF query.
    -   Send the query to the InfiniGram API (or mock API).
    -   Store the results.
4.  Print a summary of processed results to the console.
5.  Save the detailed results to the specified `output_csv_path`.

## Output CSV Format

The output CSV file (e.g., `infinigram_query_results_api.csv`) will contain the original data plus the following columns:

-   `problem`: Original problem text.
-   `answer`: Original answer text.
-   `keywords`: Comma-separated string of unique keywords extracted from the problem and answer.
-   `cnf_query_token_ids_str`: String representation of the token ID list used for the CNF query.
-   `count`: The co-occurrence count returned by InfiniGram.
-   `approx`: Boolean indicating if the InfiniGram count is approximate.
-   `notes`: Any notes regarding the processing, such as errors or API latency.

## Customizing Keyword Extraction

The `extract_keywords` function uses `spacy.load("en_core_web_sm")` and extracts all named entities (`ent.text`). You can customize this by:
-   Using a different spaCy model (e.g., `en_core_web_md` for potentially better accuracy).
-   Filtering entities by type (e.g., `ent.label_`). An example is commented within the function.
-   Adding other pre-processing or post-processing steps for keywords. 