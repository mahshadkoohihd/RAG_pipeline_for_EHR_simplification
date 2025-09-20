import os
import json
import faiss
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from sentence_transformers import SentenceTransformer

# === Setup Paths ===
base_path = "/project/perl/mk985/ishala_backup/correct_gpu_ishala/umlsProject"
input_dir_cncpt = os.path.join(base_path, "concept_input_500")
output_dir = os.path.join(base_path, "outputs")
input_dir_eval = os.path.join(base_path, "evaluation_input_15")
prompt_dir = os.path.join(base_path, "prompts_rag")
os.makedirs(prompt_dir, exist_ok=True)

# === Load spaCy and UMLS linker ===
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

# === Build Knowledge Base ===
knowledge_corpus = []
definition_lookup = []

for fname in os.listdir(output_dir):
    if fname.endswith(".txt"):
        try:
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                concepts = json.load(f)
                for item in concepts:
                    definition_raw = item.get("definition")
                    if definition_raw:
                        definition = definition_raw.strip()
                        if definition and definition not in knowledge_corpus:
                            knowledge_corpus.append(definition)
                            definition_lookup.append({
                                "definition": definition,
                                "source": f"{item.get('text', '')} ({item.get('name', '')})"
                            })
        except Exception as e:
            print(f"Could not load {fname}: {e}")

print(f"Loaded {len(knowledge_corpus)} unique UMLS definitions.")

# === Embed Knowledge Base and Build FAISS Index ===
model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
corpus_embeddings = model.encode(knowledge_corpus, convert_to_numpy=True)
faiss.normalize_L2(corpus_embeddings)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings)

# === Process Evaluation EHRs and Build Prompts ===
top_k_per_sentence = 5
similarity_threshold = 0.4

for fname in os.listdir(input_dir_eval):
    if fname.endswith(".txt"):
        input_path = os.path.join(input_dir_eval, fname)
        prompt_path = os.path.join(prompt_dir, fname.replace(".txt", "Prompt.txt"))

        with open(input_path, "r", encoding="utf-8") as f_in:
            ehr_text = f_in.read().strip()

        # === Extract Exact UMLS Concepts ===
        doc = nlp(ehr_text)
        exact_defs = []
        seen_defs = set()
        for entity in doc.ents:
            if entity._.kb_ents:
                top_cui, similarity = entity._.kb_ents[0]
                kb_entry = linker.kb.cui_to_entity[top_cui]
                definition = kb_entry.definition.strip() if kb_entry.definition else None
                if definition and definition not in seen_defs:
                    exact_defs.append(f"{entity.text} ({kb_entry.canonical_name}): {definition}")
                    seen_defs.add(definition)

        # === Split EHR into Sentences ===
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

        # === Retrieve Definitions per Sentence ===
        retrieved_defs = set()
        for sentence in sentences:
            sent_embedding = model.encode([sentence], convert_to_numpy=True)
            faiss.normalize_L2(sent_embedding)
            distances, indices = index.search(sent_embedding, top_k_per_sentence)

            for sim, i in zip(distances[0], indices[0]):
                entry = definition_lookup[i]
                if sim >= similarity_threshold and entry["definition"] not in seen_defs and entry["definition"] not in retrieved_defs:
                    retrieved_defs.add(entry["definition"])
                    print(f" ({sim:.2f}) {entry['source']}")
                else:
                    print(f" ({sim:.2f}) {entry['source']} - filtered or duplicate")

        # Format retrieved definitions with source
        formatted_retrieved = []
        for entry in definition_lookup:
            if entry["definition"] in retrieved_defs:
                formatted_retrieved.append(f"{entry['source']}: {entry['definition']}")

        # === Build Final Prompt ===
        prompt_lines = [
            "Summarize and simplify the following EHR in a way that is understandable for a 6th grade reader. I have provided a list of relevant definitions below; use them to improve the simplification.",
            "",
            ehr_text,
            "",
            "Definitions (extracted from note):"
        ] + exact_defs + [
            "",
            "Additional Definitions (retrieved by sentence similarity):"
        ] + formatted_retrieved

        with open(prompt_path, "w", encoding="utf-8") as f_prompt:
            f_prompt.write("\n".join(prompt_lines))

        print(f"? Prompt written for {fname}")
