# RAG-Enhanced EHR Text Simplification

This project prepares **LLM prompts** to simplify Electronic Health Records (EHRs) by combining **exact UMLS definitions** (via SciSpaCy entity linking) with **retrieved, definition-level context** (via Sentence-Transformers + FAISS).  
The output is a set of prompt files designed to help an LLM generate **6th-grade–readable** summaries while reducing omissions and hallucinations.

## What it does

1. **Loads a clinical NLP pipeline** (SciSpaCy `en_core_sci_scibert`) with:
   - Abbreviation expansion
   - UMLS entity linking (canonical names + definitions)

2. **Builds a lightweight knowledge base** of definitions by scanning previously extracted concept files from `outputs/` (JSON-in-.txt files).

3. **Embeds all definitions** with a biomedical Sentence-Transformer (BioBERT multi-NLI/STSB model) and **indexes them in FAISS** for efficient similarity search.

4. **Processes evaluation EHRs** (plain text in `evaluation_input_15/`):
   - Extracts **exact** UMLS definitions present in the note
   - Splits the note into sentences, **retrieves top-K semantically similar definitions** from the KB for each sentence (thresholded to avoid spam)
   - **Builds a final prompt** that includes:
     - The original EHR text
     - Exact definitions (from linking)
     - Additional retrieved definitions (from sentence similarity)

5. **Writes prompts** to `prompts_rag/*.txt` (one per input note).

---

## Project structure

~~~
umlsProject/
├─ concept_input_500/          # (optional) raw concept inputs if you keep them
├─ outputs/                    # JSON-in-.txt files with concepts & definitions (used to build KB)
│    ├─ note1.txt              # contains a JSON array of { text, name, definition } objects
│    └─ ...
├─ evaluation_input_15/        # EHR notes to simplify (one .txt per note)
│    ├─ MIMIC_0001.txt
│    └─ ...
├─ prompts_rag/                # generated prompts (output of this script)
└─ build_prompts.py            # (this script)
~~~

> **Note:** The script expects the above directories (or your custom paths) and scans `outputs/` to build the definition KB.

---

## Installation

> Python 3.10+ recommended.

~~~
# Core dependencies
pip install spacy scispacy sentence-transformers faiss-cpu

# If you have a compatible NVIDIA GPU and want GPU FAISS:
# pip install faiss-gpu

# Torch (needed by sentence-transformers). If you don't already have it:
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
~~~

### SciSpaCy model & linker data

- Install SciSpaCy’s **clinical language model** (`en_core_sci_scibert`):

~~~
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
~~~

- **UMLS linker KB**: SciSpaCy’s `scispacy_linker` with `linker_name="umls"` requires access to the UMLS KB. Follow the SciSpaCy documentation to obtain/download the KB (UMLS license may be required depending on resource).  
  If you cannot use UMLS, you can switch to other linkers (e.g., MeSH) by changing `linker_name` in the script—but the prompt content will differ.

---

## Configuration

Inside the script, adjust base paths as needed:

~~~python
base_path = "/project/perl/mk985/ishala_backup/correct_gpu_ishala/umlsProject"

input_dir_cncpt = os.path.join(base_path, "concept_input_500")  # optional
output_dir      = os.path.join(base_path, "outputs")            # concept+definition JSONs (.txt)
input_dir_eval  = os.path.join(base_path, "evaluation_input_15")# EHR notes (.txt)
prompt_dir      = os.path.join(base_path, "prompts_rag")        # where prompts will be written
~~~

Retrieval hyperparameters:

~~~python
top_k_per_sentence   = 5     # number of candidate definitions to pull per sentence
similarity_threshold = 0.4   # filter low-similarity matches
~~~

Model choices:

- **Sentence Transformer**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- **spaCy/SciSpaCy pipeline**: `en_core_sci_scibert` with `abbreviation_detector` and `scispacy_linker` (`linker_name="umls"`)

---

## Input expectations

1. **`outputs/*.txt`**  
   Each file should contain a **JSON array** of objects like:
   ~~~json
   [
     {
       "text": "AFib",
       "name": "Atrial fibrillation",
       "definition": "An irregular and often very rapid heart rhythm..."
     }
   ]
   ~~~
   The script builds the knowledge base from the unique `definition` strings across all files here.

2. **`evaluation_input_15/*.txt`**  
   Each file is a **plain-text EHR note** to be simplified.

---

## Running

~~~
python build_prompts.py
~~~

You should see logs such as:
- Number of unique definitions loaded
- Sentence-level retrieval with similarity scores
- Prompt written messages per input file

Generated prompts land in:

~~~
prompts_rag/<input_name>Prompt.txt
~~~

Each prompt has this structure:

~~~
Summarize and simplify the following EHR in a way that is understandable for a 6th grade reader...
<Original EHR text>

Definitions (extracted from note):
<term_text> (<canonical_name>): <definition>

Additional Definitions (retrieved by sentence similarity):
<source_text (canonical_name)>: <definition>
~~~

---

## Tips & troubleshooting

- **UMLS access / linker errors**:  
  If SciSpaCy can’t locate the UMLS KB, follow their docs to download/configure it.  
- **No definitions loaded**:  
  Ensure `outputs/*.txt` contains valid JSON with non-empty `"definition"` fields.  
- **Slow embedding**:  
  Cache embeddings (`np.save`) if your KB is large.  
- **GPU**:  
  Use `device="cuda"` in `SentenceTransformer`.  
- **Threshold tuning**:  
  Raise `similarity_threshold` to shorten prompts; lower it if context is sparse.

---

## Ethical & data notes

- **EHRs are sensitive**: only use properly de-identified data.  
- **UMLS**: requires licensing/credential compliance.  

---

## Citation & acknowledgements

- **SciSpaCy**: Neumann et al., *arXiv:1902.07669*  
- **UMLS**: Bodenreider, *Nucleic Acids Research 2004*  
- **Sentence-Transformers**: Reimers & Gurevych, *EMNLP 2019*  
- **FAISS**: Johnson et al., *IEEE TPAMI 2019*  
- **BioBERT model**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`

---

## License

Add your preferred license (e.g., MIT) in `LICENSE`.

---

## Maintainers

- Your Name (Affiliation, Contact)
