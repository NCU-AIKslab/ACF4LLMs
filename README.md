# Agentic Compression Framework

一個智慧型模型壓縮系統，使用 LangGraph 和 GPT-4o 自動尋找最佳壓縮策略，透過多目標優化實現自主決策。

## 概述

Agentic Compression Framework 只需最少的輸入（模型名稱 + 資料集），即可自動完成：
- 推斷模型規格與需求
- 由 LLM 自主提出壓縮策略
- 在多個基準測試上評估壓縮後的模型
- 追蹤準確度、延遲、記憶體和模型大小的 Pareto 最優解
- 根據結果迭代改進策略

## 特色功能

- **最少輸入**：只需提供模型名稱和資料集，系統會自動推斷其他設定
- **LLM 驅動決策**：使用 GPT-4o 作為協調器，自主決定壓縮策略
- **多方法支援**：AutoRound、GPTQ、INT8、AWQ、剪枝、蒸餾、LoRA/QLoRA
- **完整基準測試**：GSM8K、CommonsenseQA、TruthfulQA、HumanEval、BIG-Bench Hard
- **多目標優化**：跨多個指標追蹤 Pareto 前緣
- **LangGraph 架構**：基於狀態機的工作流程，可靠且可追蹤

## 快速開始

### 安裝

```bash
# 複製專案
git clone <repository_url>
cd Green_AI

# 建立 conda 環境
conda create -n greenai python=3.10
conda activate greenai
pip install -r requirements.txt

# （選用）安裝 GPU 量化器（需要 CUDA 工具鏈）
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit  # 更新為你的 CUDA 路徑
pip install -r requirements.quantization.txt
```

### 設定環境變數

```bash
# 必須設定 OpenAI API Key
export OPENAI_API_KEY=sk-your-api-key-here

# （選用）設定 HuggingFace Token（用於受限模型）
export HF_TOKEN=hf_your-token-here
```

## 使用範例

### 基本壓縮優化

```bash
# 最簡單的用法 - 壓縮 GPT-2 並針對 GSM8K 優化
python scripts/run_pipeline.py --model gpt2 --dataset gsm8k

# 指定更多回合以獲得更好的優化結果
python scripts/run_pipeline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset commonsenseqa \
    --episodes 10

# 互動模式 - 即時顯示進度
python scripts/run_pipeline.py \
    --model mistralai/Mistral-7B-v0.1 \
    --dataset humaneval \
    --interactive

# 自訂實驗名稱和時間預算
python scripts/run_pipeline.py \
    --model Qwen/Qwen2-7B \
    --dataset truthfulqa \
    --episodes 5 \
    --budget 4.0 \
    --experiment-name "qwen2_truthful_opt"
```

### 查看模型規格

在執行壓縮前，先查看系統自動推斷的模型規格：

```bash
python scripts/run_pipeline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --show-spec
```

輸出範例：
```
📊 Model Specification
========================
Model: meta-llama/Meta-Llama-3-8B-Instruct
Parameters: 8.03B
Architecture: LlamaForCausalLM
Original Size: 16.06 GB (FP16)

Recommended Methods: autoround, gptq, awq
Target Benchmark: gsm8k
Estimated VRAM: 20.0 GB (for quantization)
```

### 分析實驗結果

```bash
# 摘要檢視
python scripts/run_pipeline.py analyze data/experiments/your_experiment_dir

# 詳細檢視 - 顯示所有 Pareto 解
python scripts/run_pipeline.py analyze data/experiments/your_experiment_dir --format detailed

# JSON 格式輸出 - 方便程式處理
python scripts/run_pipeline.py analyze data/experiments/your_experiment_dir --format json
```

### 完整範例：壓縮 Llama-3-8B

```bash
# 1. 設定環境變數
export OPENAI_API_KEY=sk-your-api-key
export HF_TOKEN=hf_your-token  # Llama-3 需要授權

# 2. 查看模型規格
python scripts/run_pipeline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --show-spec

# 3. 執行壓縮優化（10 回合，4 小時預算）
python scripts/run_pipeline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --episodes 10 \
    --budget 4.0 \
    --interactive \
    --experiment-name "llama3_gsm8k_optimization"

# 4. 分析結果
python scripts/run_pipeline.py analyze \
    data/experiments/llama3_gsm8k_optimization \
    --format detailed
```

### 手動評估已壓縮的模型

```bash
python run_manual_eval.py \
    --model-path ./compressed_models/llama3-8b-4bit \
    --benchmark gsm8k \
    --output-dir ./eval_results
```

## 架構

### LangGraph 狀態機

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   START ──► coordinator ──┬──► quantization ──► eval   │
│                 ▲         │         │                   │
│                 │         │         ▼                   │
│                 │         │    update_state            │
│                 │         │         │                   │
│                 └─────────┴─────────┘                   │
│                 │                                       │
│                 └──► search ──┘                         │
│                 │                                       │
│                 └──► END                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 核心元件

| 元件 | 檔案 | 說明 |
|------|------|------|
| LangGraph 協調器 | `src/coordinator/langgraph_coordinator.py` | 使用 GPT-4o 的 LLM 驅動協調器 |
| 狀態管理 | `src/coordinator/state.py` | LangGraph 狀態 schema |
| 規格推斷 | `src/coordinator/spec_inference.py` | 自動推斷模型規格 |
| Pareto 前緣 | `src/coordinator/pareto.py` | 多目標優化追蹤 |
| 量化代理 | `src/agents/quantization_agent.py` | 量化工具（AutoRound、GPTQ、AWQ、INT8） |
| 評估代理 | `src/agents/evaluation_agent.py` | 基準測試評估 |

### LLM 協調器工作流程

1. **分析狀態**：GPT-4o 分析當前 Pareto 前緣和歷史記錄
2. **決定策略**：自主決定下一個壓縮策略（方法、位元數等）
3. **執行壓縮**：調用量化工具執行壓縮
4. **評估結果**：在目標基準測試上評估壓縮後的模型
5. **更新前緣**：如果結果是非支配解，加入 Pareto 前緣
6. **迭代**：根據結果決定是否繼續或終止

## 支援的模型

框架支援任何 HuggingFace 模型，針對以下模型有優化設定：

| 模型系列 | 範例 | 推薦量化方法 |
|----------|------|--------------|
| Llama | Llama-2-7B, Llama-3-8B | AutoRound, GPTQ, AWQ |
| Mistral | Mistral-7B, Mixtral-8x7B | AutoRound, GPTQ |
| Qwen | Qwen2-7B, Qwen2.5-14B | GPTQ, AWQ |
| GPT-2 | gpt2, gpt2-medium | INT8, GPTQ |

## 基準測試

| 基準測試 | 說明 | 評估指標 |
|----------|------|----------|
| GSM8K | 數學推理 | 準確率 |
| CommonsenseQA | 常識推理 | 準確率 |
| TruthfulQA | 真實性與事實性 | MC1/MC2 分數 |
| HumanEval | 程式碼生成 | pass@1 |
| BIG-Bench Hard | 多元困難任務 | 準確率 |

## 實驗目錄結構

```
data/experiments/{experiment_name}/
├── model_spec.json          # 推斷的模型規格
├── pareto_frontier.json     # Pareto 最優解
├── final_results.json       # 優化摘要
├── pareto_visualization.html # 互動式視覺化
└── episode_xxx/
    ├── strategy.json        # 壓縮策略
    └── results.json         # 評估結果
```

## CLI 選項參考

```
python scripts/run_pipeline.py [OPTIONS]

選項:
  -m, --model TEXT          HuggingFace 模型名稱或路徑 [必填]
  -d, --dataset TEXT        目標資料集 [必填]
                            可選: gsm8k, commonsenseqa, truthfulqa,
                                  humaneval, bigbench_hard
  -e, --episodes INTEGER    最大壓縮回合數 [預設: 3]
  -b, --budget FLOAT        時間預算（小時）[預設: 2.0]
  -n, --experiment-name TEXT 實驗名稱 [自動產生]
  -i, --interactive         互動模式，顯示進度更新
  --show-spec               顯示推斷的規格後退出
  -o, --output-dir TEXT     輸出目錄 [預設: data/experiments]
  -c, --config PATH         設定檔路徑（JSON 或 YAML）
  --help                    顯示說明
```

## 開發狀態

### 階段 1（MVP）✅ 完成
- 基本協調器與規格推斷
- Pareto 前緣追蹤
- CLI 介面

### 階段 2 ✅ 完成
- LangGraph 重構
- GPT-4o LLM 驅動決策
- 真實量化工具整合

### 階段 3（進行中）
- 完整基準測試實作
- 剪枝代理
- 微調代理（LoRA/QLoRA）

### 階段 4（計劃中）
- 蒸餾支援
- MLflow 整合
- Streamlit 儀表板
- Docker 部署

## 測試

```bash
# 執行基本測試
pytest tests/

# 執行端對端測試
python scripts/run_pipeline.py \
    --model gpt2 \
    --dataset gsm8k \
    --episodes 2
```

## 授權

MIT License

## 引用

如果您在研究中使用此框架，請引用：

```bibtex
@software{agentic_compression,
  title = {Agentic Compression Framework},
  year = {2024},
  author = {Your Name},
  url = {repository_url}
}
```

## 致謝

- 使用 [LangGraph](https://github.com/langchain-ai/langgraph) 建構工作流程
- 壓縮函式庫：AutoRound、GPTQ、PEFT、AWQ
- 評估由 [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) 驅動
- LLM 協調由 OpenAI GPT-4o 提供

## 聯絡方式

如有問題或需要支援，請在 GitHub 上開啟 issue。
