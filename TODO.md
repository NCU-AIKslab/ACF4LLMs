# TODO Roadmap (refreshed 2025‑11‑07)

Current focus areas for the Agentic Carbon-Efficient LLM Compression Framework. Tasks are grouped by subsystem and reference the primary files to touch.

---

## 1. Workflow & Core Engine
- [ ] **Persist LangGraph checkpoints**—swap the in-memory `MemorySaver` for SQLite/Postgres so agent restarts can resume (`src/agentic_compression/graph/workflow.py`, `graph/state.py`).  
- [ ] **Result provenance**—extend `core/metrics.py` and the UI export JSON to include tool history, agent reasoning, and evaluation config hashes.  
- [ ] **CLI polish**—wire `agentic_compression.cli` into README instructions and add smoke tests that call `agentic-compress run-rq*` with mocked evaluations.

## 2. Deep Agent & Tools
- [ ] **Offline fallback**—allow `compression_deep_agent.py` to run without Anthropic keys by swapping in a local LLM or scripted planner; document env-var driven mode switching.  
- [ ] **Tool resilience**—add retries/rate-limit handling in `agents/tracking_tool.py` and the LoRA/Distillation sub-agent tools; surface errors into `workspace/knowledge/`.  
- [ ] **Real LoRA/Distillation loops**—replace the current estimators with optional training hooks (PEFT/KD) guarded by flags so users with GPUs can enable them.

## 3. Compression & Evaluation
- [ ] **Benchmark expansion**—support MMLU, ARC (challenge/easy), and HellaSwag via `evaluation/lm_harness_adapter.py`, including per-task overrides.  
- [ ] **Carbon telemetry**—integrate ElectricityMap/WattTime in `tools/carbon_tools.py`, cache readings, and blend them with `pynvml` GPU data for precise CO₂ estimates.  
- [ ] **Additional compression strategies**—prototype mixed-precision and attention sparsification utilities under `tools/compression_tools.py`.

## 4. UI/UX & Visualization
- [ ] **Experiment runner UI**—add controls to the Streamlit pages for launching RQ1–RQ4 individually, reusing `examples/run_all_experiments.py` logic.  
- [ ] **Plot exports**—allow downloading Pareto/3D plots as SVG/PDF from `ui/components.py` + `visualizations.py`.  
- [ ] **Live dashboard**—create a compact monitoring layout showing GPU stats, carbon use, and Pareto movement in real time.

## 5. Testing & Quality
- [ ] **Tooling tests**—add `tests/test_tools/` covering quantization, pruning, carbon, and tracking tools with mocks (no GPU/API requirements).  
- [ ] **Workflow tests**—unit test the LangGraph nodes (`plan`, `evaluate`, `pareto`) using fake metrics to ensure state transitions behave.  
- [ ] **CI hygiene**—run Ruff + Black + mypy in CI and publish coverage; generate a lock file (`requirements-lock.txt` or Poetry lock) for reproducibility.

## 6. Documentation & Examples
- [ ] **Developer quick start**—write a concise guide for extending LangGraph nodes and agents (link from `README.md`).  
- [ ] **Notebooks**—ship at least one Jupyter notebook that walks through a full compression/evaluation cycle.  
- [ ] **Results gallery**—curate representative JSON output (ignored by Git) and document how to compare new runs against the gallery.

Keep this list synced with actual progress; when tasks land, move them into release notes instead of letting stale TODOs linger.***
