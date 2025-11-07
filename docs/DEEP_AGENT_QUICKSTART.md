# Deep Agent 快速開始

## 安裝

1. 確保環境已激活：
```bash
conda activate greenAI
```

2. 安裝依賴（已完成）：
```bash
pip install mlflow lm-eval
```

## 快速測試

運行測試套件驗證安裝：
```bash
python test_deep_agent.py
```

預期輸出：
```
Passed: 4/4
```

## 基本使用

### 1. 使用單個工具

#### 估算 LoRA 影響
```python
from src.agentic_compression.agents.sub_agents.lora_sub_agent import EstimateLoRAImpactTool

tool = EstimateLoRAImpactTool()
result = tool._run(
    base_model="meta-llama/Llama-2-7b-hf",
    rank=8,
)
print(result)
```

輸出：
```json
{
  "base_model": "meta-llama/Llama-2-7b-hf",
  "compression_ratio": 834.5,
  "trainable_params_percent": 0.12,
  "recommended_use_cases": ["Task-specific fine-tuning", ...]
}
```

#### 估算蒸餾效果
```python
from src.agentic_compression.agents.sub_agents.distillation_sub_agent import EstimateDistillationTool

tool = EstimateDistillationTool()
result = tool._run(
    teacher_model="meta-llama/Llama-2-7b-hf",
    compression_ratio=2.0,
)
print(result)
```

#### 記錄實驗到 MLflow
```python
from src.agentic_compression.agents.tracking_tool import LogExperimentTool

tool = LogExperimentTool()
result = tool._run(
    config={"technique": "quantization", "bits": 8},
    metrics={"accuracy": 0.654, "latency_ms": 45.3},
    model_name="llama-2-7b",
)
```

查看 MLflow UI：
```bash
mlflow ui --backend-store-uri ./mlruns
# 打開 http://localhost:5000
```

### 2. 使用 Deep Agent（需要 API key）

#### 設置 API key
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

#### 創建 agent
```python
from src.agentic_compression.agents.compression_deep_agent import create_compression_deep_agent

agent = create_compression_deep_agent(
    workspace_dir="./workspace",
    mlflow_tracking_uri="./mlruns",
)
```

#### 規劃壓縮策略
```python
plan = agent.plan_compression(
    model_name="meta-llama/Llama-2-7b-hf",
    objective="edge deployment with <2% accuracy loss",
    carbon_budget=0.05,  # 50g CO2
)

print("Agent Plan:")
print(plan['output'])
```

#### 執行實驗
```python
config = {
    "technique": "quantization",
    "bits": 8,
    "model_name": "meta-llama/Llama-2-7b-hf",
}

result = agent.execute_experiment(config, log_to_mlflow=True)
print(result['output'])
```

#### 反思和改進
```python
current_solutions = [
    {
        "config": {"quantization_bits": 8, "pruning_sparsity": 0.3},
        "metrics": {"accuracy": 0.654, "latency_ms": 45.3},
    },
    {
        "config": {"quantization_bits": 4, "pruning_sparsity": 0.5},
        "metrics": {"accuracy": 0.612, "latency_ms": 28.1},
    },
]

improvements = agent.reflect_and_improve(
    current_solutions=current_solutions,
    objective="maximize speedup with <2% accuracy loss",
)

print("Agent Suggestions:")
print(improvements['output'])
```

### 3. 整合到 LangGraph Workflow

#### 啟用 Deep Agent
```python
from src.agentic_compression.agents.workflow_integration import enable_deep_agent_workflow

# 啟用 Deep Agent（替換硬編碼規劃）
enable_deep_agent_workflow()
```

#### 運行完整優化
```python
from src.agentic_compression.graph.workflow import run_compression_optimization
from src.agentic_compression.graph.state import create_initial_state

# 創建初始狀態
state = create_initial_state(
    model_name="meta-llama/Llama-2-7b-hf",
    objective="edge deployment",
    carbon_budget=0.1,
)

# 運行優化（現在使用 Deep Agent）
result = await run_compression_optimization(
    model_name="meta-llama/Llama-2-7b-hf",
    objective="edge deployment",
    carbon_budget=0.1,
)

# 查看結果
print(f"Found {len(result['solutions'])} solutions")
print(f"Pareto frontier: {len(result['pareto_frontier'])} optimal solutions")
```

## Workspace 管理

### 查看實驗歷史
```python
import json
import os

experiments_dir = "./workspace/experiments"
for filename in os.listdir(experiments_dir):
    with open(os.path.join(experiments_dir, filename)) as f:
        exp = json.load(f)
        print(f"Experiment: {exp['id']}")
        print(f"  Config: {exp['config']}")
        print(f"  Metrics: {exp['metrics']}")
```

### 查看學習的知識
```bash
ls -la workspace/knowledge/
cat workspace/knowledge/quantization_best_practices.md
```

### 手動添加知識
```python
from src.agentic_compression.agents.compression_deep_agent import WorkspaceManager

workspace = WorkspaceManager("./workspace")
workspace.save_knowledge(
    topic="lora_best_practices",
    content="""
# LoRA Best Practices

- Use rank 8-16 for most tasks
- Target q_proj, v_proj, k_proj, o_proj for LLaMA models
- Alpha = 2 * rank (typical)
- Dropout 0.05-0.1 to prevent overfitting
"""
)
```

## 常見任務

### 任務 1: 找到最佳 8-bit 量化配置
```python
agent = create_compression_deep_agent()

task = """
Find the optimal 8-bit quantization configuration for meta-llama/Llama-2-7b-hf.
Consider different quantization methods (GPTQ, AWQ, SmoothQuant).
Estimate accuracy impact and speedup for each.
"""

result = agent.run(task)
print(result['output'])
```

### 任務 2: 比較 LoRA vs 蒸餾
```python
task = """
Compare LoRA (rank=8) vs Knowledge Distillation (50% size) for meta-llama/Llama-2-7b-hf.
Analyze:
1. Compression ratio
2. Expected accuracy impact
3. Training requirements
4. Use case recommendations
"""

result = agent.run(task)
print(result['output'])
```

### 任務 3: 碳預算優化
```python
plan = agent.plan_compression(
    model_name="meta-llama/Llama-2-7b-hf",
    objective="minimize carbon footprint while maintaining >60% accuracy",
    carbon_budget=0.05,  # 50g CO2
    constraints={"max_latency_ms": 100},
)
```

## MLflow 使用

### 啟動 MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### 查詢實驗（通過 Python）
```python
from src.agentic_compression.agents.tracking_tool import QueryExperimentsTool

tool = QueryExperimentsTool()
result = tool._run(
    filter_string="params.technique = 'quantization'",
    max_results=10,
    order_by="metrics.accuracy DESC",
)
print(result)
```

### 獲取最佳配置
```python
from src.agentic_compression.agents.tracking_tool import GetBestConfigTool

tool = GetBestConfigTool()
result = tool._run(
    metric="accuracy",
    higher_is_better=True,
    technique_filter="quantization",
)
print(result)
```

## 調試技巧

### 查看 agent 可用工具
```python
agent = create_compression_deep_agent()
print(f"Total tools: {len(agent.tools)}")
for tool in agent.tools:
    print(f"  - {tool.name}: {tool.description[:60]}...")
```

### 單獨測試工具
```python
# 找到特定工具
tool = agent.tool_dict["estimate_lora_impact"]

# 直接調用
result = tool._run(
    base_model="meta-llama/Llama-2-7b-hf",
    rank=8,
)
print(result)
```

### 檢查 workspace 狀態
```python
import os

print("Experiments:", len(os.listdir("./workspace/experiments")))
print("Knowledge:", len(os.listdir("./workspace/knowledge")))
print("Checkpoints:", len(os.listdir("./workspace/checkpoints")))
```

## 故障排除

### 問題 1: ImportError
確保已安裝所有依賴：
```bash
conda run -n greenAI pip install mlflow lm-eval langchain langchain-anthropic
```

### 問題 2: ANTHROPIC_API_KEY 未設置
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
# 或在 .env 文件中設置
```

### 問題 3: MLflow 實驗未顯示
檢查 tracking URI：
```python
import mlflow
print(mlflow.get_tracking_uri())  # 應該是 ./mlruns
```

### 問題 4: Workspace 目錄不存在
```bash
mkdir -p workspace/{experiments,knowledge,checkpoints}
```

## 下一步

1. 閱讀 `DEEP_AGENT_IMPLEMENTATION.md` 了解架構細節
2. 查看 `TODO.md` 了解未來計劃
3. 運行 `test_deep_agent.py` 驗證安裝
4. 設置 Anthropic API key 測試完整 agent 功能
5. 探索 MLflow UI 查看實驗歷史

## 相關資源

- [LangChain Deep Agents](https://blog.langchain.com/deep-agents/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
