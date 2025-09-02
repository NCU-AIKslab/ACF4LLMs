# Math 1.5B Model Training Pipeline

> ⚠️ **Work in Progress**: This project is currently under active development. The training pipeline and architecture are being debugged and optimized. The final implementation may differ from the current state.

## Overview

This repository contains a comprehensive training pipeline for a 1.5B parameter mathematical reasoning model based on Qwen2.5-Math. The pipeline implements a sophisticated multi-stage training approach combining supervised fine-tuning, reinforcement learning, and process reward modeling to create a model capable of solving complex mathematical problems using Program-of-Thought (PoT) reasoning.

## Architecture & Methodology

### Core Model
- **Base Model**: Qwen2.5-Math-1.5B (or Qwen2.5-Math-1.5B-Instruct)
- **Target Hardware**: NVIDIA A100 40GB GPU
- **Framework**: PyTorch with Flash Attention 2

### Training Pipeline Architecture

The training follows a carefully orchestrated multi-stage approach:

#### 1. **Teacher Data Synthesis & Verification**
- Uses OpenAI's GPT models as teacher to generate Program-of-Thought (PoT) solutions
- Each solution includes:
  - Python code that computes the exact answer
  - Assertions for verification
  - Final answer extraction
- **Verification Sandbox**: Executes generated code in a restricted environment with only safe imports (math, fractions, decimal, itertools, sympy)
- Auto-caches verified solutions to `/content/data/teacher_verified.jsonl` for reuse

#### 2. **Supervised Fine-Tuning (SFT)**
- Trains the base model on verified PoT examples
- Uses chat templates for instruction following
- Configuration:
  - Learning Rate: 2e-5
  - Sequence Length: 4096
  - Gradient Checkpointing enabled for memory efficiency
  - Mixed precision training (bf16)

#### 3. **Reinforcement Learning with Verifiable Rewards (RLVR)**
- Implements GRPO (Grouped Reinforcement from Preferences Optimization)
- **Reward Function**:
  - Correctness verification via symbolic math (SymPy)
  - Brevity penalty (-0.0002 * length)
  - PRM (Process Reward Model) bonus for step quality
- Configuration:
  - Learning Rate: 1e-6
  - KL coefficient: 0.02
  - Group size: 4 generations per prompt

#### 4. **Process Reward Model (PRM)**
- Sequence classification model that scores individual reasoning steps
- Trained on verified solution traces
- Used to guide solution generation and repair
- Architecture: Same base model repurposed for sequence classification with 2 labels

#### 5. **PRM-Guided Self-Evolution**
- **Best-of-N Sampling**: Generates multiple solutions and selects best based on PRM scores
- **Quality Filtering**: Only keeps solutions that are both correct AND have high PRM scores (>0.6)
- Creates enhanced training data for iterative improvement
- Implements shallow MCTS-like exploration for better coverage

#### 6. **StepCo-style Verify-Then-Revise**
- Automatic repair mechanism for incorrect solutions
- Multi-round revision with verification at each step
- Keeps solutions concise while fixing errors

#### 7. **Direct Preference Optimization (DPO)**
- Creates preference pairs: concise solutions (chosen) vs verbose ones (rejected)
- Trains model to prefer shorter, cleaner code while maintaining correctness
- Automatically strips comments and unnecessary print statements

#### 8. **Length-Aware Reinforcement Learning**
- Final refinement stage with stronger brevity penalties
- Balances solution correctness with code efficiency
- Learning Rate: 1e-6 with careful KL regularization

### Data Flow

```
GSM8K + MATH Dataset
        ↓
Teacher Synthesis (GPT)
        ↓
Verification Sandbox
        ↓
SFT Training Data
        ↓
Base Model → SFT → GRPO → PRM Training
                     ↓
              PRM-Enhanced Data
                     ↓
              DPO Training
                     ↓
            Length-Aware RL
                     ↓
            Final Model
```

## Key Innovations

1. **Verifiable Execution Environment**: All generated code is executed in a sandboxed environment to ensure correctness before training.

2. **Process Reward Modeling**: Step-level scoring allows the model to identify and improve weak reasoning steps.

3. **Self-Evolution Loop**: The model generates its own enhanced training data guided by PRM scores.

4. **Multi-Stage Optimization**: Combines different training objectives (correctness, brevity, step quality) in a principled sequence.

5. **Automatic Repair Mechanism**: Failed solutions can be iteratively refined using verification feedback.

## Evaluation

The pipeline includes comprehensive evaluation on GSM8K test set with:
- Pass@1 accuracy measurement
- Automatic solution repair for failed attempts
- Symbolic math verification for numerical equivalence
- Support for both exact matching and algebraic simplification

## Technical Details

### Dependencies
- transformers >= 4.43
- accelerate >= 0.30
- trl >= 0.9.6
- peft, datasets, bitsandbytes
- flash-attn >= 2.5.8
- sympy for mathematical verification
- lighteval >= 0.4.0 for evaluation

### Memory Optimization
- Gradient checkpointing
- Mixed precision (bf16)
- Flash Attention 2
- Efficient batch accumulation

### Safety Features
- Restricted code execution environment
- No access to file system or network in sandbox
- Deterministic random seeds for reproducibility
- Comprehensive error handling and fallbacks

## Future Enhancements (Planned)

- Integration with more diverse mathematical datasets
- Advanced MCTS for solution exploration
- Multi-modal reasoning capabilities
- Distributed training support
- Enhanced verification with formal theorem provers

## Notes

This implementation represents an experimental approach to training mathematical reasoning models with a focus on verifiable correctness and efficient code generation. The architecture combines ideas from recent research in program synthesis, process supervision, and preference learning.

The current codebase is being actively debugged and optimized. Contributions and suggestions are welcome once the initial implementation stabilizes.

## License

[To be determined]

## Contact

[Contact information to be added]

---

*This project is part of ongoing research in mathematical reasoning and program synthesis for large language models.*