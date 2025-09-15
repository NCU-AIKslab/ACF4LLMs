#!/usr/bin/env python3
"""Test script for real implementation of LLM compression agents."""

import sys
import os
sys.path.append('/mnt/Green_AI')

from llm_compressor.agents.quantization import QuantizationAgent
from llm_compressor.agents.perf_carbon import PerfCarbonAgent
from llm_compressor.agents.pruning_sparsity import PruningSparsityAgent

def test_quantization_agent():
    """Test real quantization implementation."""
    print("=" * 60)
    print("Testing Quantization Agent")
    print("=" * 60)
    
    config = {
        "calibration_samples": 128,
        "default_method": "awq"
    }
    
    agent = QuantizationAgent("quantization", config)
    
    recipe = {
        "quantization": {
            "enabled": True,
            "method": "awq",
            "bits": 8,
            "group_size": 128,
            "calibration_samples": 128
        }
    }
    
    context = {
        "model_path": "openai-community/gpt2"
    }
    
    print("Running quantization agent...")
    result = agent.execute(recipe, context)
    
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")
    
    print(f"Metrics keys: {list(result.metrics.keys())}")
    print(f"Artifacts keys: {list(result.artifacts.keys())}")
    
    return result

def test_performance_agent():
    """Test real performance measurement."""
    print("\n" + "=" * 60)
    print("Testing Performance Agent")
    print("=" * 60)
    
    config = {
        "sequence_length": 1024,
        "batch_size": 1
    }
    
    agent = PerfCarbonAgent("perf_carbon", config)
    
    recipe = {
        "quantization": {
            "enabled": True,
            "method": "awq",
            "bits": 4
        }
    }
    
    context = {
        "model_path": "openai-community/gpt2"
    }
    
    print("Running performance agent...")
    result = agent.execute(recipe, context)
    
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")
    
    # Show key metrics
    if result.success:
        metrics = result.metrics
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    return result

def test_pruning_agent():
    """Test real pruning implementation."""
    print("\n" + "=" * 60)
    print("Testing Pruning Agent")
    print("=" * 60)
    
    config = {
        "max_head_pruning": 0.3,
        "max_ffn_pruning": 0.4
    }
    
    agent = PruningSparsityAgent("pruning_sparsity", config)
    
    recipe = {
        "pruning_sparsity": {
            "enabled": True,
            "head_pruning_ratio": 0.1,
            "ffn_pruning_ratio": 0.15,
            "structured_sparsity": "2:4",
            "pruning_method": "magnitude"
        }
    }
    
    context = {
        "model_path": "openai-community/gpt2"
    }
    
    print("Running pruning agent...")
    result = agent.execute(recipe, context)
    
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")
    
    print(f"Metrics keys: {list(result.metrics.keys())}")
    
    return result

def main():
    """Run all tests."""
    print("🚀 Testing Real LLM Compression Implementation")
    print("=" * 80)
    
    # Test individual agents
    try:
        quant_result = test_quantization_agent()
        perf_result = test_performance_agent()
        prune_result = test_pruning_agent()
        
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"✅ Quantization Agent: {'PASS' if quant_result.success else 'FAIL'}")
        print(f"✅ Performance Agent:  {'PASS' if perf_result.success else 'FAIL'}")
        print(f"✅ Pruning Agent:      {'PASS' if prune_result.success else 'FAIL'}")
        
        # Check for real vs mock implementations
        print("\n🔍 Implementation Status:")
        
        if "mock" in str(quant_result.artifacts):
            print("  ⚠️  Quantization: Using mock implementation")
        else:
            print("  ✅ Quantization: Using real implementation")
            
        if "mock" in str(perf_result.metrics):
            print("  ⚠️  Performance: Using mock measurements")
        else:
            print("  ✅ Performance: Using real measurements")
            
        if "mock" in str(prune_result.metrics):
            print("  ⚠️  Pruning: Using mock implementation")
        else:
            print("  ✅ Pruning: Using real implementation")
        
        print("\n🎯 Next Steps:")
        print("  1. Run full pipeline with: make run")
        print("  2. Check execution times in logs")
        print("  3. Verify model outputs and metrics")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()