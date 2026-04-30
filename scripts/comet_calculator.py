import json
import torch
import sys

from comet import download_model, load_from_checkpoint

def calculate_backtranslation_comet(backtranslated_path, original_path):
    # 1. Load the JSON files
    with open(backtranslated_path, 'r', encoding='utf-8') as f1:
        data_hyp = json.load(f1)
    
    with open(original_path, 'r', encoding='utf-8') as f2:
        data_ref = json.load(f2)

    # 2. Extract texts
    # mt = Machine Translation (the backtranslated text)
    # ref/src = The original text before any translation occurred
    mt_texts = [obj['text'] for obj in data_hyp]
    ref_texts = [obj['text'] for obj in data_ref]

    # 3. Load the COMET model
    # 'Unbabel/wmt22-comet-da' is the state-of-the-art choice
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # 4. Format data
    # In backtranslation, the Source and Reference are identical.
    data = [
        {"src": ref, "mt": hyp, "ref": ref}
        for hyp, ref in zip(mt_texts, ref_texts)
    ]

    # 5. Run Prediction
    # Set gpus=1 if you have an NVIDIA GPU available
    device = 1 if torch.cuda.is_available() else 0
    model_output = model.predict(data, batch_size=8, gpus=device)

    # 6. Results
    print("-" * 30)
    print(f"COMET System Score: {model_output.system_score:.4f}")
    print("-" * 30)
    
    # Optional: Save individual scores back into a list to analyze outliers
    return model_output.scores

if __name__ == "__main__":
    # file1 = Backtranslated, file2 = Original
    scores = calculate_backtranslation_comet("test_dataset.json", sys.argv[1])
    
    # Quick check for the worst performing translation
    if scores:
        print(f"Lowest Segment Score: {min(scores):.4f}")
