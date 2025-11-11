# SPEC - Agentic Compression Framework for LLMs

## 整體概念：「壓縮大主管 + 專業小主管」

你要做的是一個「**壓縮總協調 Agent（Coordinator）**」+ 一堆專業小 Agent：

* **Coordinator Agent**：

  * 負責讀「任務需求 + 硬體限制 + 環境（雲端 / 邊緣）」
  * 設定壓縮目標（accuracy / latency / memory / energy / CO₂ 權重）
  * 安排「現在先試什麼方法、試多 aggressive、要不要加蒸餾」
  * 負責記錄結果、更新 Pareto frontier、最後給出「推薦壓縮配方」

* **Method Agents（專業壓縮技師）**：主要 subagents（從 MVP 到進階）。Coordinator 會向它們發出 action proposals，並收集回傳的結果／artifact。

  1. **Quantization Agent**（量化，MVP） — 執行 PTQ/QAT（AutoRound、GPTQ、INT8 等），回傳量化後 checkpoint、實際大小與兼容性資訊。
  2. **Evaluation Agent**（評估，MVP） — 提供 proxy 與完整 benchmark，回傳 accuracy/latency/memory/energy 等標準化指標向量。
  3. **Artifact / Checkpoint Manager**（MVP） — 儲存與管理模型 checkpoint、artifact 與 metadata（避免資料遺失，提供版本追蹤）。
  4. **Fine-tuning Agent**（微調，高優先） — 使用 LoRA/QLoRA 等在少量資料上恢復或提升 accuracy，並產生新的 checkpoint。
  5. **Pruning Agent**（剪枝，高優先） — 支援結構化與非結構化剪枝（LLM-Pruner、SparseGPT 等），回傳稀疏化结果與 size/throughput 變化。
  6. **Distillation Agent**（蒸餾，進階） — 在資源與 dataset 足夠時進行 teacher→student 蒸餾以降低模型尺寸或換成更小 student。
  7. **Resource Monitor Agent**（資源監控，可選） — 偵測 VRAM/GPU 型號/可用記憶體與 power，於策略執行前做 pre-check 或動態告警。
  8. **Data Manager Agent**（資料管理，可選） — 管理 calibration samples、eval 子集、資料前處理與快取，並回傳 sample paths 或訓練資料統計。
  9. **Strategy / Search Agent**（搜尋代理，可選但建議） — 負責超參與策略搜尋（BayesOpt / bandit / evolutionary），產生下一輪策略建議並更新探索分布。
 10. **Logger / Tracking Agent**（紀錄追蹤，可選） — 保存每次實驗的 metadata、metrics、Pareto 更新與可視化介面（或整合 MLflow/W&B）。

Coordinator 不直接動模型參數，而是請subagents「呼叫具體工具（AutoRound、LLM-Pruner、INC、TRL 等）」來做實際壓縮。

---

## 第一層：輸入 / 目標描述（Problem Spec）

### 1. 使用者 / 系統給的條件

為了簡化使用流程，使用者只需提供要壓縮的 model name 和要用來評估 / 蒸餾的 dataset 名稱（或 dataset 標識）。

Coordinator Agent 第一件事：
→ 讀這個簡短的輸入，推論出適當的 reward 函數與搜尋策略（例如選擇重點優化 accuracy 或 latency，並自動補足硬體偵測與預設 constraints）。

範例介面（最小）：

```json
{
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "dataset": "gsm8k"
}
```

說明：Coordinator 會根據 `model_name` 與 `dataset` 自動：

- 推估目標硬體與可用記憶體（若使用者未提供則採預設偵測或提示輸入）；
- 設定以 `dataset` 為主的評估流程（例如選取合適的 benchmark 子集與評分標準）；
- 組成內部的 reward（weighted 或 RL-style），並產生初始的壓縮策略種群（action proposals）供 Method Agents 執行；
- 若需要更細節控制，使用者可後續補充 constraints / objectives，但不是必須項。


---

## 第二層：方法空間（Action Space）
你現在只給出 `model_name` 與 `dataset`，所以 Coordinator 會先把這兩項映射到一組「推測值 / 預設約束 / 優先目標」，然後以此為基礎產生 action proposals 給各 Method Agents。

重點：Coordinator 可自動

- 根據 `model_name` 推估模型大小、參數格式與常見支援工具（例如該 model 是否適合 GPTQ / AutoRound / QAT）；
- 根據 `dataset` 推估應優先優化的指標（例如 GSM8K 偏 accuracy；對話型 dataset 可能偏 latency/throughput）；
- 對常見硬體做快速檢查（若無明確硬體輸入，採用預設策略或提示使用者）；
- 為每個 method 自動填充合理的超參範圍與預設值，使用者可選擇覆寫。

以下仍列出各 Method Agent 的 action space，但請注意：在實作中許多欄位可標記為 optional，Coordinator 會在策略生成時自動補齊或搜尋。

### 1. Quantization Agent 的可選動作

- 套件 / 方法（Coordinator 會根據 model compatibility 建議優先選項）：

  - AutoRound（2/3/4-bit weight-only PTQ）
  - GPTQ / AWQ（4-bit）
  - INT8 / 混合精度 PTQ / QAT（例如透過 INC）

- 可控超參（可標記為 optional；Coordinator 可給出預設或搜尋範圍）：

  - bit_width（2,3,4,8）
  - per_tensor / per_channel
  - 是否做 SmoothQuant / 梯度感知 clipping
  - 是否搭配 small FT（QAT）

動作例子（Coordinator 可能只要求 minimal fields，其他由系統補齊）：

```json
{
  "method": "AutoRound",
  "bit_width": 4
  // group_size, calib_samples 可由 Coordinator 補齊或在 search 中優化
}
```

### 2. Pruning Agent

- 套件（依 model 與目標自動推薦）：

  - LLM-Pruner（結構化剪枝）
  - SparseGPT（非結構化、高稀疏）
  - INC pruning API

- 超參（Coordinator 可設定探索範圍）：

  - sparsity_ratio（例如 0.1–0.5）
  - prune_granularity（head / channel / block）
  - one_shot vs iterative

### 3. Distillation Agent

- 套件（在 dataset 規模與資源允許下啟用）：

  - HF Transformers + TRL 的 Generalized KD Trainer
  - INC distillation

- 超參：

  - KD_temperature
  - loss 組合（KD + CE + 可選 RL-style reward）
  - student_size（例如 8B → 3B）

Coordinator 會判斷是否啟動 KD（例如當 dataset 足夠大或 accuracy 是主要目標時）。

### 4. Fine-tuning Agent

- 用 LoRA / QLoRA 或小規模 FT 來恢復或提升壓縮後的性能：

  - learning_rate
  - epochs
  - lora_rank / alpha

在 MVP 階段，Coordinator 可把 fine-tuning 當作可選步驟（例如僅在 accuracy drop 超過門檻時啟動）。

---

## 第三層：流程（Multi-Agent Workflow）

流程仍是「多輪實驗迭代 loop」，但起點改為以最小輸入（`model_name` + `dataset`）為基礎，讓 Coordinator 自動推論缺少的細節並驅動整個實驗迭代。

1. Initialization（啟動）

   * Coordinator 先做快速探測與推論：

     - 根據 `model_name` 推估模型大小、參數格式、相容的工具（例：GPTQ、AutoRound、INT8 支援）與大致 VRAM 要求。
     - 根據 `dataset` 推估要優先優化的指標（例如 GSM8K 偏 accuracy；對話型 dataset 可能偏 latency/throughput）。
     - 若硬體或資源資訊缺失，Coordinator 可用預設值、嘗試自動偵測，或產生簡短的互動式提示要求使用者補充。

   * 產生初始策略種群（initial population / seed proposals）：Coordinator 會用上述推論建立數個起始候選（例如保守/中間/激進三種），每個候選包含 quantization/pruning/distill/ft 的 minimal fields。

2. 每一輪迭代（Episode）

   (a) Strategy Proposal — Coordinator

   * 以過往結果（歷史紀錄 / Pareto frontier）和內部 reward（由 `model_name`+`dataset` 推導）作為條件，產生或挑選一個策略：

     - 若資源受限或 dataset 小，Coordinator 會偏向 cheaper 操作（e.g. INT8、少量 LoRA）；若 accuracy 是首要，會偏向 KD / more aggressive quantization+FT 組合。
     - Coordinator 可決定是否在 proposal 中把某些欄位留作 `auto`（由 Method Agents 在執行時再搜尋或自動填補）。
     - 若缺少關鍵約束（例如最大 VRAM），Coordinator 可主動向使用者提問以避免無效嘗試。

   (b) Apply Compression — Method Agents

   * 各 Method Agent 接收策略（可能包含 auto placeholders）並執行：

     - Quantization Agent：呼叫最佳相容工具並回報實際產出（size, quantization artifacts）。
     - Pruning Agent：視資源決定 one-shot 或 iterative，並回報稀疏化結果。
     - Distillation / Fine-tuning Agent：在 dataset 與資源允許下啟動，並生成 student/checkpoint。

   * 若某步驟因資源不足或工具不相容而無法執行，該策略會被標記為 invalid 或打上大負分（hard constraint 處理）。

   (c) Evaluate — Evaluation Agent

   * 評估流程由 Coordinator 根據 `dataset` 與推論出的目標指標決定：

     - 使用 dataset 的子集或快速 proxy 基準先做快速驗證（fast filter），對過濾過的候選再跑完整 benchmark。
     - 蒐集的指標通常包含 accuracy、latency（ms/token）、memory（實際 VRAM 使用）、模型檔案大小與能源 proxy（或實測 GPU power log）。

   * 輸出標準化結果向量（可直接送入 reward）：

     ```json
     {
       "strategy_id": "...",
       "accuracy": 0.92,
       "latency_ms": 130,
       "memory_gb": 6,
       "energy_j_per_1k_tokens": 35,
       "valid": true
     }
     ```

   (d) Update & Learn — Coordinator

   * 把新結果加入多目標優化框架：

     - 更新 Pareto frontier、保存非支配解；
     - 用 reward（weighted 或 RL）更新策略產生器的先驗分布（例如 bandit/RL/Bayesian 更新）；
     - 決定下一輪是 exploit（在好解附近微調）或 explore（跳到新組合）。

3. Stopping & Output

   * 停止條件典型包含：預算（GPU 小時）耗盡；達成使用者硬性條件（若使用者指定）；或策略收斂（若多輪都無明顯改進）。

   * 最終輸出：

     - 最佳 N 個 Pareto 解與對應策略描述（含哪些欄位由 Coordinator auto 填充）；
     - 壓縮後模型 checkpoint（或 checkpoint 存放路徑）與完整實驗紀錄（metrics、log、environment）；
     - 一份 human-readable 報告，說明為何選擇該組合與後續可行的改進建議（例如若使用者想更激進的壓縮，需補充的資源或 dataset）。

補充：

- 為了 UX，Coordinator 可以在第一輪或必要時啟動短對話式互動（prompt the user）以確認關鍵假設（例如是否允許 accuracy drop 超過某門檻），這樣就能在 minimal input 與安全約束間取得平衡。

---

## 第四層：Reward / 評分設計（核心）

你必須把多目標變成「Agent 能理解的 reward」，簡單版可以用 weighted sum：

[
R = \alpha \cdot Acc_{norm} - \beta \cdot Latency_{norm} - \gamma \cdot Mem_{norm} - \delta \cdot CO2_{norm}
]

* 你可以讓 user 的 `objectives` 決定 (\alpha, \beta, \gamma, \delta)。
* 再加上一些 **hard constraint**：

  * 若 accuracy drop > threshold → 直接給大負分。
  * 若 VRAM 超過硬體 → 方案無效，不納入 frontier。

這樣 Coordinator 就可以用 RL / bandit / Bayesian optimization 等方法，在策略空間裡搜尋。

---

## 第五層：實作落地建議

目標：提供一個可執行的 MVP 路徑，從最小輸入（`model_name` + `dataset`）出發，實作一個輕量的 Coordinator + 少量 Method Agents（quantization + optional fine-tune），並能跑快速的驗證評估。

產出物（建議最小集合）

- `src/coordinator.py` — 負責把 `model_name`/`dataset` 轉成 internal spec、產生初始策略並驅動 episode loop。
- `src/agents/quantization.py` — 包裝 AutoRound / GPTQ / INT8 執行介面（或 mock 介面，MVP 可先 mock 執行以便開發）。
- `src/agents/finetune.py` — 用 LoRA/QLoRA 做小規模微調（MVP 可選做）。
- `src/evaluation.py` — 提供快速 proxy 評估（subset accuracy、latency estimator、vram check），並輸出標準化結果向量。
- `scripts/run_pipeline.py` — CLI 入口：接受 `--model` `--dataset` 與資源上限（budget、max_vram 等）。
- `tests/test_infer_spec.py` — 單元測試：驗證 Coordinator 的推論邏輯（model→推估結果，dataset→primary objective）。

MVP 工作流程（技術步驟）

1. 建立 minimal input parser（CLI / REST）：接收 `model_name` 與 `dataset`，並可選 `--budget` / `--max-vram`。

2. implement `infer_spec(model_name, dataset)`：一個 deterministic helper，負責推估模型大小、相容工具、primary objective 與預設 constraints（示例代碼見下）。

3. 根據 infer_spec 生成 3 個 seed strategies（conservative / balanced / aggressive），每個 strategy 只包含必要欄位（例如 quantization.method、bit_width、do_finetune:boolean）。

4. 執行策略：先用 proxy evaluation（小數據子集 + latency estimator）快速篩選，再對 top-K 做完整評估。

5. 更新簡單的 Pareto set（可只用 list + non-dominated filter），重複直到 budget 或次數上限。

6. 最後輸出最佳策略與對應的 artifacts（checkpoint 路徑、metrics JSON、human-readable report）。

簡單的 Coordinator helper（範例）

```python
def infer_spec(model_name: str, dataset: str) -> dict:
  """Infer a small spec from model_name and dataset for the MVP coordinator.

  Returns a dict with keys: model_size_gb, preferred_quant, primary_objective, default_constraints
  """
  spec = {}
  # naive model size inference based on name tokens (MVP heuristic)
  if any(x in model_name.lower() for x in ("7b", "8b", "8b-instruct", "8b-instruct")):
    spec["model_size_gb"] = 20
  elif any(x in model_name.lower() for x in ("3b", "3b-")):
    spec["model_size_gb"] = 8
  else:
    spec["model_size_gb"] = 40  # conservative default

  # dataset -> primary objective
  if dataset.lower().startswith("gsm") or "math" in dataset.lower():
    spec["primary_objective"] = "accuracy"
  else:
    spec["primary_objective"] = "latency"

  # preferred quant methods heuristics
  spec["preferred_quant"] = ["gptq", "autoround"] if spec["model_size_gb"] > 10 else ["int8", "autoround"]

  spec["default_constraints"] = {"max_accuracy_drop": 0.02, "max_vram_gb": None}
  return spec

```

快速 CLI 範例（使用者可直接跑）

```bash
python scripts/run_pipeline.py --model "meta-llama/Meta-Llama-3-8B-Instruct" --dataset gsm8k --budget 2h
```

驗證與測試建議

- 單元測試：`infer_spec` 應在常見 model/dataset 上回傳合理的 primary objective 與預設 constraints。
- 集成測試：對一個非常小的模型或 mock model（或用小型學生模型）跑 end-to-end pipeline，檢查 pipeline 是否能完成一輪 episode 並輸出 metrics JSON。
- 快速迴歸：在 CI 中增加一個 smoke test（只跑 proxy eval）以確保變更不會破壞主流程。

依賴（MVP 建議）

- core: Python 3.10+
- libraries: `transformers`, `datasets`, `numpy`, `psutil`（檢查 VRAM/memory), `pytest`（測試）
- 可選（若採用實體量化工具）: `bitsandbytes`, `accelerate`, `gptq` 對應套件

逐步擴展建議（後續優化）

- 把 `infer_spec` 替換成一個小型 ML model 或 rules engine，以更精準地推估硬體/工具相容性。
- 引入一個輕量的 search module（bayesian optimization / bandit）來替換 naive seed + mutate 的策略生成。
- 新增一個 metadata store（簡單的 SQLite 或 JSONL）來保存每次 experiment 的環境、註解與 artifacts，便於重現與分析。

結語

這一層的目的是把設計推向可執行的原型：用最小的接口（`model_name` + `dataset`）啟動一個可重現的 pipeline，先以 proxy 評估保護資源，然後逐步放寬到完整工具鏈（AutoRound/GPTQ/LoRA/KD）。如果你要，我可以：

1) 在 repo 中新增 `src/coordinator.py` 與 `scripts/run_pipeline.py` 的初始骨架並提交一個 PR；
2) 或先產出一個更完整的 `infer_spec` 實作與對應單元測試，並在本機跑一次 smoke test。