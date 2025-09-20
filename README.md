# RAG_pipeline_for_EHR_simplification
This repo builds prompts for clinical text simplification by (1) extracting exact UMLS concepts/definitions with SciSpaCy, (2) retrieving additional semantically relevant definitions using BioBERT embeddings + FAISS, and (3) assembling final prompts that guide an LLM to produce 6th-grade–level summaries of discharge notes.
