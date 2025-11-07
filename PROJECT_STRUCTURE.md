# 📂 Project Structure (v2.1)

最新的目錄整理，對齊目前倉庫實際狀態與 v2.1 的程式碼分層。此文件可作為快速導覽，方便追蹤主要程式碼、測試、實驗與長期記憶資產。

---

## 📁 目錄快照

```
Green_AI/
├── README.md
├── requirements.txt
├── app.py
├── quick_test.py
├── test_real_evaluation.py
├── test_deep_agent.py
├── AGENT.md
├── DEEP_AGENT_IMPLEMENTATION.md
├── DEEP_AGENT_QUICKSTART.md
├── REAL_EVALUATION_IMPLEMENTATION.md
├── NO_API_KEY_NEEDED.md
├── SUMMARY.md
├── TODO.md
├── PROJECT_STRUCTURE.md  ← 本文件
│
├── docs/
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── EXECUTE_THIS.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── CLAUDE.md
│   └── TODO.md
│
├── examples/
│   ├── run_all_experiments.py
│   ├── simple_optimization.py
│   ├── use_without_api_key.py
│   └── __init__.py
│
├── src/
│   └── agentic_compression/
│       ├── __init__.py
│       ├── cli.py
│       ├── agents/
│       ├── core/
│       ├── evaluation/
│       ├── graph/
│       ├── inference/
│       ├── optimization/
│       ├── tools/
│       ├── ui/
│       └── visualization/
│
├── tests/
│   ├── test_core/
│   └── test_optimization/
│
├── workspace/
│   ├── README.md
│   ├── checkpoints/
│   ├── experiments/
│   └── knowledge/
│
├── mlruns/            # MLflow tracking artifacts
├── .claude/           # Deep Agent local settings
├── .pytest_cache/     # pytest cache（可清除）
├── .ruff_cache/       # Ruff cache（可清除）
└── .git/ …            # Git metadata
```

---

## 🚪 Entry Points & Scripts
- `app.py`：新的根目錄 Streamlit wrapper，可直接 `streamlit run app.py`。
- `quick_test.py`：匯入與環境檢查的煙霧測試。
- `test_real_evaluation.py`：跑完真實模型壓縮 + lm-eval 基準的整合測試（預設 quick mode）。
- `examples/run_all_experiments.py`：一次性觸發 RQ1–RQ4 的研究流程。
- `streamlit run src/agentic_compression/ui/app.py`：啟動 Web UI；若想保留舊指令，也可自行建立根目錄 wrapper `app.py`。
- `tests/`：使用 `pytest` 執行單元與研究邏輯測試，建議配合 `PYTHONPATH=$(pwd)/src`。

---

## 🧱 Source Package (`src/agentic_compression`)

| 子模組 | 說明 |
| --- | --- |
| `core/` | 核心資料結構，例如 `CompressionConfig`、`EvaluationMetrics`、Pareto 演算法。 |
| `optimization/` | 研究問題 (RQ1–RQ4) 的策略引擎；`agent_driven.py` 為 v2.1 主力。 |
| `tools/` | 量化、剪枝、碳排監測與 lm-eval 工具，供代理與工作流呼叫。 |
| `evaluation/` | `BenchmarkRunner` 與 `lm_harness_adapter`，處理真實基準執行。 |
| `graph/` | LangGraph 狀態機與 workflow 入口 (`workflow.py`)。 |
| `agents/` | 深層代理（Anthropic Deep Agent、子代理工具、追蹤工具）。 |
| `inference/` | 模型載入、量化、剪枝實作細節。 |
| `ui/` | Streamlit app (`app.py` + components/visualizations/utils)。 |
| `visualization/` | Pareto 與多維圖表產生器（Plotly）。 |

---

## 📚 Documentation Sets
- 根目錄：針對代理/深度實驗的專題說明 (`AGENT.md`, `DEEP_AGENT_*.md`, `REAL_EVALUATION_IMPLEMENTATION.md`, `SUMMARY.md`, `TODO.md`)。
- `docs/`：面向使用者與開發者的指南：
  - `README.md`: 文檔索引
  - `QUICKSTART.md`, `EXECUTE_THIS.md`: 中文快速上手與指令清單
  - `IMPLEMENTATION_SUMMARY.md`, `CLAUDE.md`: 架構與開發者指南
  - `TODO.md`: 跨文件任務佇列

---

## 🧪 Tests & Experiment Artifacts
- `tests/test_core/`：對 `core` 元件（config、metrics、pareto）的單元測試。
- `tests/test_optimization/`：驗證代理驅動優化流程。
- `mlruns/`：MLflow run 資料夾（可清空或加入 `.gitignore` 保持乾淨）。
- `workspace/`：Deep Agent 長期記憶（`experiments/`, `knowledge/`, `checkpoints/`）與 `README` 說明，可備份或納入版本控制以追蹤代理學習。

---

## ♻️ 建議的清理策略
1. **產物分層**：`mlruns/`、`eval_results/`、`model_cache/` 及 `workspace/` 子資料夾（checkpoints/、experiments/、knowledge/）已加入 `.gitignore`，可放心保留本地結果而不污染版本控制。
2. **快取清理**：`.pytest_cache/` 與 `.ruff_cache/` 可在需要時安全刪除，避免污染差異。
3. **入口統一**：`app.py` 已作為 Streamlit wrapper，既可沿用 `streamlit run app.py`，也可直接執行 `streamlit run src/agentic_compression/ui/app.py`。
4. **檔案分門別類**：若 `AGENT.md`、`DEEP_AGENT_*.md` 需要更高可發現性，可考慮移到 `docs/agents/` 子資料夾並更新引用（目前保留在根目錄以利深度代理開發記事）。

---

## ✅ 快速檢查清單
- [ ] `src/agentic_compression/` 為唯一可分發的 Python 套件來源。
- [ ] 實驗腳本皆位於 `examples/` 或專用 `tests/` 下。
- [ ] 文檔分成「根目錄研究筆記」與 `docs/` 專用指南。
- [ ] 產物/快取獨立於 `workspace/`、`mlruns/`、`.pytest_cache/`、`.ruff_cache/`。
- [ ] `PROJECT_STRUCTURE.md` 已與實際檔案同步，可作為未來增減檔案的更新基準。
