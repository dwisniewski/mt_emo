import json
import torch
import sys
import pandas as pd
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer, util

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [obj['text'] for obj in json.load(f)]

def run_analysis(hyp1_path, hyp2_path, ref_path):
    # 1. Load the JSON files
    texts_hyp1 = load_data(hyp1_path)
    texts_hyp2 = load_data(hyp2_path)
    texts_ref = load_data(ref_path)

    if not (len(texts_hyp1) == len(texts_hyp2) == len(texts_ref)):
        raise ValueError("Error: All input files must have the same number of examples.")

    # --- PART 1: COMET Analysis ---
    print("Starting COMET analysis...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    device = 0 if torch.cuda.is_available() else None # COMET uses 0/1 for GPU or None for CPU
    print(device)
    
    data1 = [{"src": r, "mt": h, "ref": r} for h, r in zip(texts_hyp1, texts_ref)]
    data2 = [{"src": r, "mt": h, "ref": r} for h, r in zip(texts_hyp2, texts_ref)]

    out1 = comet_model.predict(data1, batch_size=8, gpus=device)
    out2 = comet_model.predict(data2, batch_size=8, gpus=device)

    comet_results = []
    for i in range(len(texts_ref)):
        comet_results.append({
            "index": i,
            "reference": texts_ref[i],
            "hyp1": texts_hyp1[i],
            "hyp2": texts_hyp2[i],
            "comet_score1": out1.scores[i],
            "comet_score2": out2.scores[i],
            "comet_diff": abs(out1.scores[i] - out2.scores[i])
        })

    # Sort and Save COMET CSV
    df_comet = pd.DataFrame(comet_results).sort_values(by="comet_diff", ascending=False)
    df_comet.to_csv("comet_comparison.csv", index=False, encoding='utf-8-sig')
    print("Saved COMET results to comet_comparison.csv")


    # --- PART 2: Semantic Similarity (Sentence-BERT) ---
    print("\nStarting Semantic Similarity analysis (Hyp1 vs Hyp2)...")
    # 'all-MiniLM-L6-v2' is fast and accurate for similarity tasks
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings
    emb1 = sbert_model.encode(texts_hyp1, convert_to_tensor=True)
    emb2 = sbert_model.encode(texts_hyp2, convert_to_tensor=True)

    # Compute cosine similarities
    # util.cos_sim returns a matrix; we want the diagonal (i-th hyp1 vs i-th hyp2)
    cosine_scores = util.cos_sim(emb1, emb2).diagonal()

    sim_results = []
    for i in range(len(texts_ref)):
        similarity = cosine_scores[i].item()
        sim_results.append({
            "index": i,
            "hyp1": texts_hyp1[i],
            "hyp2": texts_hyp2[i],
            "semantic_similarity": similarity,
            "semantic_difference": 1 - similarity # Higher means more semantically different
        })

    # Sort by difference (Highest difference/Lowest similarity first)
    df_sim = pd.DataFrame(sim_results).sort_values(by="semantic_difference", ascending=False)
    df_sim.to_csv("similarity_comparison.csv", index=False, encoding='utf-8-sig')
    print("Saved Similarity results to similarity_comparison.csv")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python comet_and_sim_scorer.py <hyp1.json> <hyp2.json> <reference.json>")
    else:
        run_analysis(sys.argv[1], sys.argv[2], sys.argv[3])
