# TODO List - Agentic Compression Framework v2.1

This document tracks active work items now that all placeholder agents and fake tests have been removed.

---

## 🔴 Critical

1. **CLI distribution**
   - Wire the existing `agentic_compression.cli` module into packaging/CI so `agentic-compress` is tested before release.
   - Add instructions in the README describing the CLI workflow.

2. **Dependency locking**
   - Produce a reproducible lock file (e.g., `requirements-lock.txt` or `poetry.lock`) so evaluation runs cannot silently drift.

---

## 🟡 High Priority

### Automated Testing
- Add coverage for LangChain tools (`tests/test_tools/`) to verify quantization/pruning/LoRA helpers without touching GPUs.
- Add workflow tests to ensure `agentic_compression.graph.workflow` state machines recover from failures.
- Create high-level tests for `run_all_experiments.py` that mock evaluation results to keep runtime short.

### Research Question Execution
- Parameterize `examples/run_all_experiments.py` so model/threshold settings can be provided via CLI flags.
- Snapshot representative experiment outputs into `results/` (ignored by Git) for regression comparisons.

### Deep Agent Hardening
- Provide offline fallbacks so `compression_deep_agent.py` can degrade gracefully when Anthropic keys are absent.
- Add rate-limit/backoff handling and surface better telemetry in `workspace/` logs.

---

## 🟢 Medium Priority

### Documentation & Guides
- Add docstring examples for public APIs.
- Publish a short “developer quick start” focused on extending LangGraph nodes.
- Create at least one Jupyter notebook walking through a complete compression run.

### Visualization Enhancements
- Animate Pareto frontier evolution over iterations.
- Allow exporting charts to SVG/PDF for paper figures.
- Add a compact dashboard layout for live monitoring.

### Carbon & Telemetry
- Integrate a regional carbon-intensity API (ElectricityMap or WattTime) and merge it with local GPU telemetry.
- Store historical carbon data in `workspace/knowledge/` for longitudinal analysis.

---

## 🔵 Future Enhancements

### Additional Compression Strategies
- Investigate mixed-precision (layer-wise) and attention sparsification.
- Prototype vocabulary pruning or adapter dropping for small-footprint models.

### Code Quality
- Run Ruff + Black in CI and enforce type checking with `mypy`.
- Add richer exception types for workflow nodes and tool failures.

---

Keep this list in sync whenever major features land so contributors do not reintroduce placeholders or fake coverage.
