#!/usr/bin/env python3
"""
Test orchestrated system WITHOUT carbon tracking for fair accuracy comparison.
This gives us the true baseline performance of the quantized system.
"""

import torch
import time
import json
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# Set environment variable for proper encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import EXACT same parsing functions
from src.eval.gsm8k_data import extract_answer, compare_answers

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_orchestrated_baseline():
    """Test orchestrated system WITHOUT carbon tracking for pure performance baseline."""
    logger.info("🎯 Starting ORCHESTRATED BASELINE TEST (NO CARBON)")
    logger.info("=" * 80)
    logger.info("📊 Goal: Get true accuracy of quantized system without overhead")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Load GSM8K dataset (same 200 questions)
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]
    num_questions = 200
    test_questions = test_data.select(range(num_questions))
    logger.info(f"Testing with {num_questions} GSM8K questions")

    # Load model with EXACT same quantization as orchestrated system
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    logger.info(f"Loading QUANTIZED model: {model_name}")

    # Use EXACT same BitsAndBytes config as orchestrator
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    if device == 'cuda':
        memory_used = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU Memory: {memory_used:.1f} MB (quantized)")

    # Test WITHOUT carbon tracking
    logger.info("🏃 Processing 200 questions WITHOUT carbon overhead...")

    results = []
    correct_count = 0
    start_time = time.time()

    for i, example in enumerate(tqdm(test_questions, desc="Processing baseline")):
        try:
            question = example["question"]
            full_gsm8k_answer = example["answer"]

            # Same prompt format as orchestrated system
            prompt = f"Problem: {question}\\nSolution:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)

            if device == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Same generation parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.01,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode and evaluate
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Use same parsing
            is_correct = compare_answers(response, full_gsm8k_answer)
            if is_correct:
                correct_count += 1

            results.append({
                'question_id': i,
                'correct': is_correct,
                'response': response[:100],  # Truncated for storage
            })

            # Log progress
            if (i + 1) % 20 == 0:
                current_acc = correct_count / (i + 1) * 100
                logger.info(f"🎯 Progress: {i+1}/200 ({current_acc:.1f}% accuracy)")

        except Exception as e:
            logger.warning(f"Error on question {i+1}: {e}")
            results.append({'question_id': i, 'correct': False, 'error': str(e)})

    total_time = time.time() - start_time

    # Final results
    accuracy = correct_count / len(results) * 100

    final_summary = {
        "test_type": "ORCHESTRATED_BASELINE_NO_CARBON",
        "model_name": model_name,
        "model_type": "orchestrated_quantized_baseline",
        "status": "COMPLETED",
        "timestamp": time.time(),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_performance": {
            "questions_correct": correct_count,
            "accuracy_percentage": accuracy,
            "total_test_time_minutes": total_time / 60,
            "questions_processed": len(results)
        },
        "memory_usage": {
            "gpu_memory_mb": memory_used if device == 'cuda' else 0,
            "quantization": "BitsAndBytes NF4 4-bit",
            "precision": "float16"
        },
        "system_info": {
            "architecture": "orchestrated_quantized",
            "quantization_method": "bnb_nf4_double_quant",
            "carbon_tracked": False,
            "overhead": "minimal_no_carbon_tracking"
        }
    }

    # Display results
    logger.info("\\n" + "🎯 ORCHESTRATED BASELINE RESULTS" + "=" * 50)
    logger.info(f"🏆 Accuracy: {accuracy:.1f}% ({correct_count}/200)")
    logger.info(f"⏱️ Time: {total_time/60:.1f} minutes")
    logger.info(f"💾 Memory: {memory_used:.1f} MB")

    # Save results
    results_file = "orchestrated_baseline_final.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\\n💾 Results saved to {results_file}")
    logger.info("🎯 ORCHESTRATED BASELINE TEST COMPLETE!")

    return True, final_summary

if __name__ == "__main__":
    try:
        success, results = test_orchestrated_baseline()
        if success:
            print("\\n🎯 ORCHESTRATED BASELINE SUCCESS!")
            print("📊 Pure quantized performance measured!")
        else:
            print("\\n❌ Baseline measurement failed.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()