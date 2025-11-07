# Agent Architecture Guide

This repository treats “agents” as modular compression specialists that are
coordinated by a LangGraph workflow instead of a classic multi-process loop.
The notes below explain the moving pieces, how they interact, and what needs
to happen when you introduce new agent types.

## Control Plane
- **Coordinator Service** – `src/agentic_compression/graph/workflow.py` builds the
  LangGraph state machine (plan → evaluate → pareto → refine). It decides which
  compression configs to try, calls evaluation tools, and tracks carbon/iteration
  budgets. There is no separate runner: invoking
  `run_compression_optimization()` spins up the whole workflow.
- **Agent Registry Placeholder** – `src/agentic_compression/agents/coordinator.py`
  and `agents/base.py` provide a hook for a future registry. Today they mainly
  document the intended interfaces; real work happens through LangChain tools.
- **State Definition** – `src/agentic_compression/graph/state.py` declares the
  shared `CompressionState` (messages, configs, solutions, budgets, errors). All
  nodes in the workflow mutate this state instead of passing raw Python objects.

## Operational Plane
- **Quantization Agent** – `tools/compression_tools.py::QuantizationTool` wraps
  `RealQuantizer` for NF4/INT8 bitsandbytes flows (`inference/quantizer.py`). It
  exposes compression metrics for planners and handles actual model loading.
- **Pruning Agent** – `PruningTool` + `RealPruner` (unstructured and 2:4 / 4:8
  structured) estimate speedups and perform pruning inside evaluation runs.
- **Distillation Agent** – `DistillationTool` simulates student-layer trade-offs.
  Training loops are not implemented yet; the tool only returns planning hints.
- **Memory/KV Agent** – `KVCacheTool` reduces context length or switches cache
  strategies to trade accuracy for latency/memory. This is the closest thing to
  a “memory optimization agent” today.
- **Missing Agents** – The current tree has no LoRA/token optimization agents.
  See `TODO.md` for the implementation backlog.

## Evaluation & Feedback Loop
- **Benchmark Runner** – `evaluation/benchmark_runner.py` plus
  `evaluation/lm_harness_adapter.py` run GSM8K, TruthfulQA, CommonsenseQA,
  HumanEval, and BigBench using lm-eval-harness. Each run produces a full
  `EvaluationMetrics` object (`core/metrics.py`).
- **Pareto Selection** – `core/pareto.py` compares solutions across accuracy,
  latency, memory, energy, and CO₂. The workflow’s `compute_pareto` node chooses
  the “best” solution depending on the objective (accuracy / carbon / balanced).

## Extending the System
1. **Define the capability** in `tools/` (either a LangChain `BaseTool` or a
   concrete agent subclass of `BaseCompressionAgent`). Keep I/O JSON-serializable.
2. **Hook it into the workflow**. Update `plan_optimization()` to seed configs or
   add a new node that calls the tool. Use `state["strategy_results"]` to persist
   intermediate outputs between nodes.
3. **Update evaluation**. If the new agent modifies the model differently
   (e.g., LoRA adapters), extend `tools/evaluation_tools.py` so
   `evaluate_config_full()` can apply that modification before benchmarking.
4. **Surface results** in the UI/CLI. Streamlit pages and the CLI expect the
  `best_solution` and `pareto_frontier` schema from `EvaluationMetrics`; make
  sure the new agent writes data in the same format.

## Debug Tips
- Run `python quick_test.py` to verify key dependencies (Transformers, lm-eval,
  bitsandbytes, GPU availability) before troubleshooting agent code.
- Use the `tests/test_optimization/` suite to confirm the workflow still converges
  after adding agents or modifying planning heuristics.
- When experiments misbehave, enable logging via the `LOG_LEVEL` env var or by
  adjusting the `logging` configuration inside individual modules.

This document should give contributors enough context to reason about how the
“agents” fit together even though most behavior currently lives in LangGraph,
LangChain tools, and evaluation utilities instead of long-running actor objects.
