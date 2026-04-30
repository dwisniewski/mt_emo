import json
import os
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Configuration ---
INPUT_FILE = "test_dataset.json"
INPUT_LANGUAGE = "English"

TARGET_LANGUAGES = {
    "fr": "French",
    "de": "German",
    "pl": "Polish",
    "es": "Spanish",
    "it": "Italian"
}

# 2026 Sub-9B Benchmark Models
MODELS_TO_TEST = [
   # "utter-project/EuroLLM-9B-Instruct",
    #"Qwen/Qwen3-8B,
    #"google/gemma-2-9b-it"
    #"stelterlab/EuroLLM-9B-Instruct-AWQ",
    "solidrust/gemma-2-9b-it-AWQ",
    #"Qwen/Qwen3-8B-AWQ",
    #"Orion-zhen/aya-expanse-8b-AWQ"
    #"ModelSpace/GemmaX2-28-9B-v0.1"
    #"haoranxu/ALMA-7B-R"
    #"Unbabel/Tower-Plus-9B"
]

# Evaluation Scenarios
SCENARIOS = {
    "basic": " Return only the translated text.",
    "emotional": " Return only the translated text. Please focus on preserving the emotions, tone, and intensity of the original text."
}

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
    return [e['text'] for e in data]
    # return list(data.keys())

def run_evaluation():
    all_data = load_data(INPUT_FILE)
    english_texts = [e['text'] for e in all_data]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    for model_id in MODELS_TO_TEST:
        model_name_clean = model_id.split('/')[-1]
        print(f"\n{'='*60}\nLOADING MODEL: {model_id}\n{'='*60}")

        # Load tokenizer separately to apply the correct Chat Template for each model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm = LLM(model=model_id, trust_remote_code=True, gpu_memory_utilization=0.9, quantization='awq_marlin')

        for scenario_name, extra_instruction in SCENARIOS.items():
            for lang_code, lang_name in TARGET_LANGUAGES.items():
                print(f"Running [{scenario_name}] scenario for {lang_name}...")
                
                # Constructing the prompts using the model's native chat template
                prompts = []
                for idx, text in enumerate(english_texts):
                    full_instruction = f"Translate the following text from {INPUT_LANGUAGE} to {lang_name}.{extra_instruction}".strip()
                    
                    messages = [
                        {"role": "user", "content": f"{full_instruction}\n\nText: {text}"}
                    ]
                    # Apply template and remove the trailing assistant header if necessary (vLLM handles this well)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)

                # Batch generate
                outputs = llm.generate(prompts, sampling_params)
                
                results = []
                for i, output in enumerate(outputs):
                    results.append({
                        #"model": model_id,
                        #"scenario": scenario_name,
                        #"target_language": lang_name,
                        #"english_text": english_texts[i],
                        "text": output.outputs[0].text.strip().split("\n\n")[0],
                        "labels": all_data[i]['labels']
                    })

                # Save: results_EuroLLM-9B-Instruct_emotional_pl.json
                output_path = f"results_{model_name_clean}_{scenario_name}_{lang_code}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

        # Clear GPU Memory
        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {model_id}. GPU memory cleared.")

if __name__ == "__main__":
    run_evaluation()

