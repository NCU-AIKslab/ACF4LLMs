#!/usr/bin/env python3
"""
Uncompressed GSM8K test WITH CARBON TRACKING for green AI comparison.
Uses exact same parsing as orchestrated system.
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

# Import CodeCarbon for emissions tracking
try:
    from codecarbon import EmissionsTracker
    CARBON_AVAILABLE = True
except ImportError:
    CARBON_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_qwen_uncompressed_with_carbon_tracking():
    """Test uncompressed Qwen WITH CARBON TRACKING for green AI comparison."""
    logger.info("🌱 Starting UNCOMPRESSED vs ORCHESTRATED GREEN AI Comparison Test")
    logger.info("=" * 80)
    logger.info("🔋 Carbon emission tracking ENABLED")
    logger.info("📝 Using EXACT same parsing methods as orchestrated system")
    logger.info("🎯 Goal: Complete green AI comparison with CO2 metrics")
    logger.info("=" * 80)

    # Initialize carbon tracker
    carbon_tracker = None
    if CARBON_AVAILABLE:
        carbon_tracker = EmissionsTracker(
            project_name="uncompressed_gsm8k_carbon_test",
            output_dir="./carbon_logs",
            save_to_file=True,
            log_level="WARNING"
        )
        carbon_tracker.start()
        logger.info("✅ Carbon tracking started successfully")
    else:
        logger.error("❌ CodeCarbon not available - cannot track emissions!")
        return False, None

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
        if carbon_tracker:
            carbon_tracker.stop()
        return False, None

    # Select same 200 questions as orchestrated test
    num_questions = min(200, len(test_data))
    test_questions = test_data.select(range(num_questions))
    logger.info(f"Testing with {num_questions} GSM8K questions")
    logger.info("🔄 SAME 200 questions as orchestrated system for fair comparison")

    # Load UNCOMPRESSED model (same model as orchestrated system)
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    logger.info(f"Loading UNCOMPRESSED model: {model_name}")
    logger.info("⚡ NO quantization - full precision model for carbon comparison")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load WITHOUT quantization
        if device == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,  # Use float16 but NO quantization
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
        if carbon_tracker:
            carbon_tracker.stop()
        return False, None

    # Test with CARBON TRACKING
    logger.info(f"\\nTesting {num_questions} GSM8K Problems (UNCOMPRESSED + CARBON TRACKING)")
    logger.info("-" * 70)
    logger.info("🌱 Carbon emissions being tracked throughout evaluation")

    results = []
    total_inference_time = 0
    correct_count = 0
    error_count = 0
    results_file = "qwen_uncompressed_carbon_comparison.json"

    start_test_time = time.time()

    # Process questions with carbon tracking
    for i, example in enumerate(tqdm(test_questions, desc="Processing questions")):
        try:
            question = example["question"]
            full_gsm8k_answer = example["answer"]
            expected_answer = extract_answer(full_gsm8k_answer)

            # Create prompt EXACTLY like orchestrated system
            prompt = f"Problem: {question}\\nSolution:"

            # Tokenize with SAME settings as orchestrated system
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            if device == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with SAME parameters as orchestrated system
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

            # Store result
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
                'model_type': 'uncompressed_carbon_tracked'
            }
            results.append(result)

            # Log progress every 10 questions
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / (i + 1) * 100
                avg_time = total_inference_time / (i + 1) * 1000
                elapsed_minutes = (time.time() - start_test_time) / 60

                logger.info(f"\\n🔋 Progress: {i+1}/{num_questions} ({(i+1)/num_questions*100:.1f}%)")
                logger.info(f"📊 Accuracy: {current_accuracy:.1f}% ({correct_count}/{i+1})")
                logger.info(f"⏱️ Avg time: {avg_time:.0f}ms | Elapsed: {elapsed_minutes:.1f}min")

        except Exception as e:
            logger.warning(f"Error processing question {i+1}: {e}")
            error_count += 1

    total_test_time = time.time() - start_test_time

    # Stop carbon tracking
    emissions = 0.0
    if carbon_tracker:
        emissions = carbon_tracker.stop()
        logger.info(f"🌱 Carbon tracking completed - Emissions: {emissions:.6f} kg CO2")

    # Final results
    logger.info("\\n" + "=" * 80)
    logger.info("🌱 UNCOMPRESSED GREEN AI COMPARISON TEST RESULTS")
    logger.info("=" * 80)

    accuracy = correct_count / len(results) * 100 if results else 0
    avg_time = total_inference_time / len(results) * 1000 if results else 0
    total_minutes = total_test_time / 60

    # Create comprehensive summary with carbon metrics
    final_summary = {
        "test_type": "QWEN_UNCOMPRESSED_CARBON_COMPARISON_FINAL",
        "model_name": model_name,
        "model_type": "uncompressed_carbon_tracked",
        "parsing_method": "orchestrated_system_exact_match",
        "carbon_tracking": "enabled",
        "comparison_purpose": "green_ai_comparison_with_carbon_metrics",
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
            "prompt_format": "Problem: {question}\\\\nSolution:"
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
        "carbon_metrics": {
            "total_emissions_kg_co2": emissions or 0.0,
            "total_emissions_g_co2": (emissions or 0.0) * 1000,
            "emissions_per_question_g": ((emissions or 0.0) * 1000) / len(results) if results else 0,
            "carbon_efficiency_accuracy_per_g_co2": accuracy / ((emissions or 0.001) * 1000)  # Avoid division by zero
        },
        "memory_usage": {
            "gpu_memory_mb": memory_used if device == 'cuda' else 0,
            "quantization": "None - UNCOMPRESSED",
            "precision": "float16" if device == 'cuda' else "float32"
        },
        "detailed_results_complete": results,
        "green_ai_metrics": {
            "carbon_tracked": True,
            "energy_efficiency": "unoptimized_baseline",
            "memory_efficiency": "no_compression",
            "carbon_comparison_ready": True
        }
    }

    # Display final results
    logger.info(f"🎯 Model: {model_name} (UNCOMPRESSED)")
    logger.info(f"📊 Questions Processed: {len(results)}/{num_questions}")
    logger.info(f"✅ Correct Answers: {correct_count}")
    logger.info(f"🏆 Final Accuracy: {accuracy:.2f}%")
    logger.info(f"❌ Errors: {error_count}")
    logger.info(f"⏱️ Average Time: {avg_time:.0f}ms per question")
    logger.info(f"⏰ Total Time: {total_minutes:.2f} minutes")
    logger.info(f"🌱 Total CO2 Emissions: {(emissions or 0.0) * 1000:.3f}g")
    logger.info(f"🌿 CO2 per Question: {((emissions or 0.0) * 1000) / len(results):.3f}g" if results else "0g")

    if device == 'cuda':
        logger.info(f"💾 GPU Memory: {memory_used:.1f} MB")

    # Save final results
    final_results_file = "qwen_uncompressed_carbon_comparison_final.json"
    try:
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        file_size = Path(final_results_file).stat().st_size / (1024 * 1024)
        logger.info(f"\\n💾 Final results saved to {final_results_file}")
        logger.info(f"📁 File size: {file_size:.2f} MB")
        logger.info(f"✅ ALL DATA COMPLETE with carbon metrics!")
        logger.info(f"🌱 Ready for GREEN AI comparison with orchestrated results")

    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

    logger.info(f"\\n🌱 UNCOMPRESSED GREEN AI COMPARISON TEST FINISHED!")
    logger.info(f"📋 Results ready for carbon-aware GitHub submission")

    return True, final_summary

if __name__ == "__main__":
    try:
        success, results = test_qwen_uncompressed_with_carbon_tracking()
        if success:
            print("\\n🌱 UNCOMPRESSED GREEN AI COMPARISON SUCCESS!")
            print("📊 Results ready with carbon emission metrics!")
            print("💾 Carbon-tracked data saved with complete analysis!")
        else:
            print("\\n❌ Uncompressed carbon comparison test failed.")
    except Exception as e:
        print(f"Test error: {str(e)}")
        import traceback
        traceback.print_exc()