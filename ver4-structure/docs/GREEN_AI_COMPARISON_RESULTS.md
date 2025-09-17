# Green AI Comparison: Orchestrated vs Uncompressed GSM8K Evaluation

## Executive Summary

This study compares the carbon footprint and performance of quantized (orchestrated) vs uncompressed model evaluation on the GSM8K mathematical reasoning dataset using Qwen/Qwen2.5-Math-1.5B.

## Key Findings

**🌱 Carbon Efficiency Results:**
- **Orchestrated System**: 47% lower CO2 emissions (220g vs 417g)
- **Memory Optimization**: 63% reduction in GPU memory usage (1,101 MB vs 2,944 MB)
- **Performance**: Maintained comparable accuracy (27.5% vs 25.0%)

## Test Configurations

### Orchestrated System (Quantized)
- **Model**: Qwen/Qwen2.5-Math-1.5B
- **Quantization**: BitsAndBytes NF4 4-bit with double quantization
- **Memory Usage**: 1,101 MB GPU memory
- **Carbon Tracking**: Direct measurement with CodeCarbon

### Uncompressed System (Baseline)
- **Model**: Qwen/Qwen2.5-Math-1.5B (full precision)
- **Quantization**: None
- **Memory Usage**: 2,944 MB GPU memory
- **Carbon Tracking**: Direct measurement with CodeCarbon

## Detailed Results

| Metric | Orchestrated System | Uncompressed System | Improvement |
|--------|-------------------|-------------------|-------------|
| **Accuracy** | 27.5% (55/200) | 25.0% (50/200) | +2.5% |
| **Total CO2 Emissions** | 219.9g | 417.1g | **-47.3%** |
| **CO2 per Question** | 1.10g | 2.09g | **-47.4%** |
| **GPU Memory** | 1,101 MB | 2,944 MB | **-62.6%** |
| **Test Duration** | 42.1 minutes | 39.3 minutes | -6.6% |
| **Carbon Efficiency** | 0.125 acc/g | 0.060 acc/g | **+108%** |

## Green AI Impact

### Carbon Footprint Reduction
- **47% CO2 reduction** demonstrates significant environmental benefit
- **197g CO2 savings per 200 questions** scales meaningfully for large deployments
- Carbon efficiency more than doubled (0.125 vs 0.060 accuracy per gram CO2)

### Resource Optimization
- **63% memory reduction** enables deployment on lower-spec hardware
- Maintained performance while reducing environmental impact
- Demonstrates feasibility of green AI optimization

## Technical Implementation

### Orchestrated System Architecture
```
ModelOrchestrator → BnBQuantizer → GSM8K Evaluator → Carbon Tracker
```

### Evaluation Methodology
- Same 200 GSM8K questions for both systems
- Identical parsing and evaluation logic
- Direct carbon measurement with CodeCarbon
- GPU memory monitoring throughout evaluation

## File Structure
```
orchestrated_carbon_measurement_final.json    # Quantized system results
qwen_uncompressed_carbon_comparison_final.json # Baseline system results
orchestrated_carbon_measurement.py            # Quantized test script
qwen_uncompressed_carbon_comparison.py        # Baseline test script
```

## Conclusion

The orchestrated quantization system demonstrates clear green AI benefits:
- **Environmental**: 47% reduction in CO2 emissions
- **Resource**: 63% reduction in memory requirements
- **Performance**: Maintained accuracy with slight improvement

This validates quantization as an effective green AI optimization strategy for mathematical reasoning tasks.

## Data Availability

All raw test results, carbon measurements, and evaluation scripts are included in this repository for reproducibility.