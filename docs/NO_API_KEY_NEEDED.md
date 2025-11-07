# 無需 API Key 使用指南

## 🎉 好消息：大部分功能都不需要 API key！

你可以直接使用所有壓縮工具和實驗追蹤功能，**無需任何 API key**。

## 什麼需要 API key？什麼不需要？

### ❌ 需要 API key 的功能（僅 1 個）

**Deep Agent 自主規劃** - 使用 LLM 進行智能決策：
- `agent.plan_compression()` - 自動規劃壓縮策略
- `agent.execute_experiment()` - 自主執行實驗
- `agent.reflect_and_improve()` - 反思和改進

**為什麼需要？** 因為 agent 使用大語言模型作為"大腦"來推理和決策。

### ✅ 不需要 API key 的功能（全部工具）

以下所有功能都**完全獨立**，不需要任何外部 API：

#### 1. LoRA/PEFT 工具 ✓
- 配置 LoRA 適配器
- 估算壓縮影響
- 載入 PEFT 模型

#### 2. 知識蒸餾工具 ✓
- 設置 teacher-student 配置
- 估算蒸餾效果
- 比較模型大小

#### 3. 量化和剪枝工具 ✓
- 量化配置計算
- 剪枝策略規劃
- KV cache 優化

#### 4. MLflow 實驗追蹤 ✓
- 記錄實驗
- 查詢歷史
- 分析最佳配置

#### 5. Workspace 記憶管理 ✓
- 保存實驗結果
- 存儲知識文檔
- 管理檢查點

## 快速演示

運行完整示例（無需 API key）：

```bash
python examples/use_without_api_key.py
```

輸出：
```
🎉 Deep Agent 工具使用示例 - 無需 API Key

示例 1: 估算 LoRA 壓縮效果 ✓
示例 2: 規劃知識蒸餾 ✓
示例 3: 實驗追蹤和分析 ✓
示例 4: Workspace 記憶管理 ✓
示例 5: 壓縮技術比較報告 ✓

✅ 所有示例完成！
```

## 使用場景

### 場景 1: 研究和分析（無需 API key）

```python
from src.agentic_compression.agents.sub_agents.lora_sub_agent import EstimateLoRAImpactTool

# 快速估算不同配置
tool = EstimateLoRAImpactTool()

for rank in [4, 8, 16, 32]:
    result = tool._run(base_model="meta-llama/Llama-2-7b-hf", rank=rank)
    print(f"Rank {rank}: {result}")
```

### 場景 2: 實驗追蹤（無需 API key）

```python
from src.agentic_compression.agents.tracking_tool import LogExperimentTool, QueryExperimentsTool

# 記錄實驗
log_tool = LogExperimentTool()
log_tool._run(
    config={"technique": "quantization", "bits": 8},
    metrics={"accuracy": 0.654, "latency_ms": 45.3},
)

# 查詢歷史
query_tool = QueryExperimentsTool()
results = query_tool._run(max_results=10)
```

### 場景 3: 知識管理（無需 API key）

```python
from src.agentic_compression.agents.compression_deep_agent import WorkspaceManager

workspace = WorkspaceManager("./workspace")

# 保存學習的最佳實踐
workspace.save_knowledge(
    topic="quantization_tips",
    content="# 8-bit 量化最適合大多數場景..."
)

# 讀取知識
tips = workspace.load_knowledge("quantization_tips")
```

## 如果你想使用完整的 Deep Agent

你有三個選項：

### 選項 1: 使用 Anthropic Claude（需要 API key）

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

```python
from src.agentic_compression.agents.compression_deep_agent import create_compression_deep_agent

agent = create_compression_deep_agent()
plan = agent.plan_compression(model_name="...", objective="...")
```

**獲取 API key**: https://console.anthropic.com/

### 選項 2: 使用 OpenAI（如果你有）

修改 `compression_deep_agent.py`:

```python
from langchain_openai import ChatOpenAI

self.llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### 選項 3: 使用本地模型（完全免費）🎉

**安裝 Ollama**:
```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 下載模型
ollama pull llama2
```

**使用本地 agent**:
```python
from src.agentic_compression.agents.local_agent import create_local_compression_agent

# 完全在本地運行！
agent = create_local_compression_agent(model_name="llama2")
plan = agent.plan_compression(model_name="...", objective="...")
```

支持的模型：
- `llama2` (7B, 13B, 70B)
- `mistral` (7B)
- `codellama` (7B, 13B, 34B)
- `mixtral` (8x7B)
- 更多：https://ollama.ai/library

## 推薦工作流

### 開始階段（無需 API key）
1. 使用工具估算不同壓縮技術的效果
2. 記錄實驗到 MLflow
3. 比較結果並選擇最佳配置

### 生產階段（可選使用 agent）
1. 安裝 Ollama 使用本地 agent（免費）
2. 或者獲取 Anthropic API key（付費）
3. 讓 agent 自動規劃和優化

## 成本對比

| 方案 | 成本 | 功能 |
|------|------|------|
| **僅使用工具** | 免費 | 估算、追蹤、分析（90% 功能）|
| **本地 Ollama** | 免費 | 工具 + 自主規劃（100% 功能）|
| **Anthropic Claude** | ~$0.003/1K tokens | 工具 + 高級規劃（最佳效果）|
| **OpenAI GPT-4** | ~$0.01/1K tokens | 工具 + 高級規劃 |

## 實際測試結果

我們運行了完整測試套件：

```bash
$ python examples/use_without_api_key.py

✅ 示例 1: 估算 LoRA 壓縮效果 - PASS
✅ 示例 2: 規劃知識蒸餾 - PASS
✅ 示例 3: 實驗追蹤和分析 - PASS
✅ 示例 4: Workspace 記憶管理 - PASS
✅ 示例 5: 壓縮技術比較報告 - PASS

所有功能正常，無需 API key！
```

## 總結

### 核心觀點

1. **90% 的功能不需要 API key**
   - 所有工具都是獨立的
   - 實驗追蹤完全本地
   - Workspace 記憶系統本地存儲

2. **Deep Agent 是可選的增強功能**
   - 提供自主規劃和決策
   - 可以使用本地模型（免費）
   - 或者付費使用雲端 API（更好的效果）

3. **靈活的部署選項**
   - 開始時：只用工具（免費）
   - 進階：Ollama 本地 agent（免費）
   - 生產：雲端 API（付費，最佳）

### 建議路徑

```
第 1 週: 使用工具探索和實驗（無需 API key）
  ↓
第 2-3 週: 安裝 Ollama，嘗試本地 agent（仍然免費）
  ↓
生產部署: 根據需求決定是否使用雲端 API
```

## 問題？

- 工具使用問題：查看 `examples/use_without_api_key.py`
- 本地 agent 設置：查看 `src/agentic_compression/agents/local_agent.py`
- 完整文檔：查看 `DEEP_AGENT_QUICKSTART.md`

**記住：你現在就可以開始使用，不需要等待任何 API key！** 🚀
