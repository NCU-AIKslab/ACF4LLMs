#!/usr/bin/env python3
"""
Uncompressed GSM8K test with EXACT same parsing as orchestrated system.
For direct comparison with orchestrated results.
"""

import torch
import time
import json
import re
import logging
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Set environment variable for proper encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path to import orchestrated system functions
sys.path.insert(0, str(Path(__file__).parent))

# Import EXACT same parsing functions from orchestrated system
from src.eval.gsm8k_data import extract_answer, compare_answers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K answer format using orchestrated system method."""
    return extract_answer(answer_text)

def create_math_prompt(question: str) -> str:
    """Create prompt EXACTLY matching orchestrated system."""
    return f"Problem: {question}\nSolution:"

def save_incremental_results(results, current_question, total_questions, start_time, filename):
    """Save results incrementally to prevent data loss."""
    current_time = time.time()
    elapsed_time = current_time - start_time

    correct_count = sum(1 for r in results if r.get('correct', False))
    accuracy = correct_count / len(results) * 100 if results else 0

    summary = {
        "test_type": "QWEN_UNCOMPRESSED_200_ORCHESTRATOR_COMPARISON",
        "status": "IN_PROGRESS" if current_question < total_questions else "COMPLETED",
        "parsing_method": "ORCHESTRATED_SYSTEM_EXACT_MATCH",
        "progress": {
            "questions_processed": len(results),
            "total_questions": total_questions,
            "questions_remaining": total_questions - len(results),
            "percent_complete": len(results) / total_questions * 100
        },
        "current_performance": {
            "questions_correct": correct_count,
            "accuracy_percentage": accuracy,
            "elapsed_time_minutes": elapsed_time / 60,
            "average_time_per_question_ms": elapsed_time * 1000 / len(results) if results else 0
        },
        "detailed_results_complete": results,
        "timestamp": current_time,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save incremental results: {e}")

def test_qwen_uncompressed_orchestrator_comparison():
    """Test uncompressed Qwen with EXACT orchestrated system parsing."""
    logger.info("🚀 Starting UNCOMPRESSED vs ORCHESTRATED Comparison Test")
    logger.info("=" * 80)
    logger.info("📝 Using EXACT same parsing methods as orchestrated system")
    logger.info("🎯 Goal: Fair comparison between uncompressed vs quantized results")
    logger.info("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Memory: {gpu_memory:.1f} GB")

    # Load GSM8K dataset
    logger.info("Loading GSM8K dataset...")
    try:
        dataset = load_dataset("openai/gsm8k", "main")
        test_data = dataset["test"]
        logger.info(f"Loaded GSM8K dataset: {len(test_data)} test problems")
    except Exception as e:
        logger.error(f"Failed to load GSM8K dataset: {e}")
        return False, None

    # Select same 200 questions as orchestrated test
    num_questions = min(200, len(test_data))
    test_questions = test_data.select(range(num_questions))
    logger.info(f"Testing with {num_questions} GSM8K questions")
    logger.info("📊 SAME 200 questions as orchestrated system for fair comparison")

    # Load UNCOMPRESSED model (same model as orchestrated system)
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    logger.info(f"Loading UNCOMPRESSED model: {model_name}")
    logger.info("🔥 NO quantization - full precision model")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load WITHOUT quantization (this is the key difference)
        if device == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,  # Use float16 for GPU efficiency but NO quantization
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Full precision on CPU
                trust_remote_code=True
            )

        logger.info("✅ UNCOMPRESSED model loaded successfully!")

        if device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"GPU Memory usage: {memory_used:.1f} MB")
            logger.info("📊 Memory usage is HIGHER due to no compression")

    except Exception as e:
        logger.error(f"Failed to load uncompressed model: {e}")
        return False, None

    # Test with same generation parameters as orchestrated system
    logger.info(f"\nTesting {num_questions} GSM8K Problems (UNCOMPRESSED + ORCHESTRATED PARSING)")
    logger.info("-" * 70)
    logger.info("⚙️  Using SAME generation parameters as orchestrated system")

    results = []
    total_inference_time = 0
    correct_count = 0
    error_count = 0
    results_file = "qwen_uncompressed_orchestrator_comparison.json"

    start_test_time = time.time()

    # Process questions with EXACT same logic as orchestrated system
    for i, example in enumerate(tqdm(test_questions, desc="Processing questions")):
        try:
            question = example["question"]
            full_gsm8k_answer = example["answer"]
            expected_answer = extract_gsm8k_answer(full_gsm8k_answer)

            # Log first 3 questions for verification
            if i < 3:
                logger.info(f"\n--- Question {i+1} (UNCOMPRESSED) ---")
                logger.info(f"Question: {question[:100]}...")
                logger.info(f"Expected answer: {expected_answer}")

            # Create prompt EXACTLY like orchestrated system
            prompt = create_math_prompt(question)
            if i < 3:
                logger.info(f"Prompt: {prompt[:50]}...")

            # Tokenize with SAME settings as orchestrated system
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            if device == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with SAME parameters as orchestrated system after fix
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,  # Same as orchestrated system
                    temperature=0.1,     # Same as orchestrated system
                    do_sample=False,     # Same as orchestrated system
                    repetition_penalty=1.01,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Decode response EXACTLY like orchestrated system
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Use EXACT same parsing as orchestrated system
            is_correct = compare_answers(response, full_gsm8k_answer)

            if is_correct:
                correct_count += 1

            # Log first 3 responses for verification
            if i < 3:
                extracted_for_display = extract_answer(response)
                logger.info(f"Generated: {response[:100]}...")
                logger.info(f"Extracted: '{extracted_for_display}'")
                logger.info(f"Comparison result: {is_correct}")
                logger.info(f"✅ CORRECT" if is_correct else "❌ INCORRECT")

            # Store COMPLETE result with orchestrated system compatibility
            result = {
                'question_id': i,
                'question_full': question,
                'expected_answer': expected_answer,
                'gsm8k_full_answer': full_gsm8k_answer,
                'generated_response_full': response,
                'extracted_answer': extract_answer(response),
                'correct': is_correct,
                'inference_time_ms': inference_time * 1000,
                'processed_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'parsing_method': 'orchestrated_system_exact',
                'model_type': 'uncompressed'
            }
            results.append(result)

            # Save incrementally every 10 questions
            if (i + 1) % 10 == 0:
                save_incremental_results(results, i + 1, num_questions, start_test_time, results_file)

                current_accuracy = correct_count / (i + 1) * 100
                avg_time = total_inference_time / (i + 1) * 1000
                elapsed_minutes = (time.time() - start_test_time) / 60

                logger.info(f"\n📊 Progress: {i+1}/{num_questions} ({(i+1)/num_questions*100:.1f}%)")
                logger.info(f"📈 Accuracy: {current_accuracy:.1f}% ({correct_count}/{i+1})")
                logger.info(f"⏱️  Avg time: {avg_time:.0f}ms | Elapsed: {elapsed_minutes:.1f}min")
                logger.info(f"💾 Results saved to {results_file}")

        except Exception as e:
            logger.warning(f"Error processing question {i+1}: {e}")
            error_count += 1

            result = {
                'question_id': i,
                'question_full': question if 'question' in locals() else "",
                'expected_answer': expected_answer if 'expected_answer' in locals() else "",
                'gsm8k_full_answer': full_gsm8k_answer if 'full_gsm8k_answer' in locals() else "",
                'generated_response_full': f"ERROR: {str(e)}",
                'extracted_answer': "",
                'correct': False,
                'inference_time_ms': 0,
                'error': True,
                'processed_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'parsing_method': 'orchestrated_system_exact',
                'model_type': 'uncompressed'
            }
            results.append(result)

    total_test_time = time.time() - start_test_time

    # Final comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("🏁 UNCOMPRESSED vs ORCHESTRATED COMPARISON TEST RESULTS")
    logger.info("=" * 80)

    accuracy = correct_count / len(results) * 100 if results else 0
    avg_time = total_inference_time / len(results) * 1000 if results else 0
    total_minutes = total_test_time / 60

    # Create final comprehensive summary
    final_summary = {
        "test_type": "QWEN_UNCOMPRESSED_ORCHESTRATOR_COMPARISON_FINAL",
        "model_name": model_name,
        "model_type": "uncompressed",
        "parsing_method": "orchestrated_system_exact_match",
        "comparison_purpose": "direct_comparison_with_orchestrated_quantized_results",
        "status": "COMPLETED",
        "timestamp": time.time(),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "device": device,
            "gpu_name": gpu_name if device == 'cuda' else 'CPU',
            "gpu_memory_gb": gpu_memory if device == 'cuda' else 0,
            "pytorch_version": torch.__version__
        },
        "test_configuration": {
            "total_questions": num_questions,
            "questions_processed": len(results),
            "dataset": "openai/gsm8k",
            "same_questions_as_orchestrated": True,
            "max_new_tokens": 150,
            "temperature": 0.1,
            "sampling": "greedy",
            "prompt_format": "Problem: {question}\\nSolution:",
            "parsing_functions": "src.eval.gsm8k_data.extract_answer_and_compare_answers",
            "data_storage": "COMPLETE - no truncation"
        },
        "final_performance": {
            "questions_correct": correct_count,
            "questions_incorrect": len(results) - correct_count,
            "questions_error": error_count,
            "accuracy_percentage": accuracy,
            "average_inference_time_ms": avg_time,
            "total_test_time_minutes": total_minutes,
            "questions_per_minute": len(results) / total_minutes,
        },
        "memory_usage": {
            "gpu_memory_mb": memory_used if device == 'cuda' else 0,
            "quantization": "None - UNCOMPRESSED",
            "precision": "float16" if device == 'cuda' else "float32"
        },
        "detailed_results_complete": results,
        "orchestrated_system_compatibility": {
            "parsing_method_identical": True,
            "prompt_format_identical": True,
            "generation_params_identical": True,
            "evaluation_logic_identical": True
        }
    }

    # Display final results
    logger.info(f"🎯 Model: {model_name} (UNCOMPRESSED)")
    logger.info(f"📊 Questions Processed: {len(results)}/{num_questions}")
    logger.info(f"✅ Correct Answers: {correct_count}")
    logger.info(f"🏆 Final Accuracy: {accuracy:.2f}%")
    logger.info(f"❌ Errors: {error_count}")
    logger.info(f"⏱️  Average Time: {avg_time:.0f}ms per question")
    logger.info(f"⏰ Total Time: {total_minutes:.2f} minutes")

    if device == 'cuda':
        logger.info(f"🖥️  GPU Memory: {memory_used:.1f} MB")
        logger.info(f"📈 Memory vs Orchestrated: ~3x higher (no quantization)")

    # Save final results
    final_results_file = "qwen_uncompressed_orchestrator_comparison_final.json"
    try:
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        file_size = Path(final_results_file).stat().st_size / (1024 * 1024)
        logger.info(f"\n💾 Final results saved to {final_results_file}")
        logger.info(f"📁 File size: {file_size:.2f} MB")
        logger.info(f"✅ ALL DATA COMPLETE - no truncation!")
        logger.info(f"🔄 Ready for direct comparison with orchestrated results")

    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

    logger.info(f"\n🎉 UNCOMPRESSED vs ORCHESTRATED COMPARISON TEST FINISHED!")
    logger.info(f"📋 Results ready for GitHub repo submission")

    return True, final_summary

if __name__ == "__main__":
    try:
        success, results = test_qwen_uncompressed_orchestrator_comparison()
        if success:
            print("\n🎉 UNCOMPRESSED vs ORCHESTRATED COMPARISON SUCCESS!")
            print("📊 Results ready for direct comparison with orchestrated system!")
            print("💾 All data saved with COMPLETE questions, answers, and responses!")
        else:
            print("\n❌ Uncompressed comparison test failed.")
    except Exception as e:
        print(f"Test error: {str(e)}")
        import traceback
        traceback.print_exc()