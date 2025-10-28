# 🚀 Quick Start Guide

快速開始使用 Agentic Carbon-Efficient LLM Compression Framework v2.0

---

## 📦 安裝

### 方法 1: 使用 requirements.txt（推薦）

```bash
# 克隆倉庫
cd /path/to/Green_AI

# 安裝所有依賴
pip install -r requirements.txt
```

### 方法 2: 僅安裝核心依賴

```bash
# 最小安裝（僅運行 Streamlit UI）
pip install streamlit plotly pandas
```

---

## 🎯 啟動 Streamlit Web UI

**最簡單的方式：**

```bash
streamlit run app.py
```

瀏覽器會自動打開 `http://localhost:8501`

---

## 🖥️ 使用 Web UI

### 主頁面
- 查看框架概述
- 了解 4 個研究問題 (RQ1-4)
- 閱讀快速開始指南

### Page 1: 🚀 Quick Optimization
1. 在側邊欄配置參數：
   - 選擇模型 (Gemma, LLaMA)
   - 設置量化位數 (4/8/16/32)
   - 設置剪枝稀疏度 (0.0-0.7)
   - 設置碳預算 (1-20 kg CO₂)
   - 設置準確度閾值 (0.80-0.99)
2. 點擊 **▶️ Run Optimization**
3. 查看結果：
   - Pareto 前沿可視化
   - 最佳解決方案
   - 基準測試準確度
4. 下載 JSON 結果

### Page 2: 📊 Advanced Visualization
1. 選擇研究問題標籤頁：
   - **RQ1**: 動態 vs 靜態壓縮比較
   - **RQ3**: 權重方案分析
   - **RQ4**: 環境適應測試
2. 配置實驗參數
3. 點擊 **▶️ Run Experiment**
4. 查看互動圖表和關鍵發現

### Page 3: 🔬 Experiment Comparison
1. 使用側邊欄添加多個實驗
2. 每個實驗可以有不同配置
3. 查看並排比較表格
4. 查看疊加的 Pareto 前沿圖
5. 下載所有結果

### Page 4: 🎯 Interactive 3D Explorer
1. 配置探索設置
2. 點擊 **🔍 Explore Solution Space**
3. 查看 3D 互動可視化：
   - **3D Pareto 前沿**（可旋轉、縮放）
   - **平行坐標圖**（可過濾）
   - **雷達圖**（多目標性能）
4. 分析碳影響

---

## 💻 編程方式使用

### 示例 1: 簡單優化

```python
import asyncio
from agentic_compression.graph.workflow import run_compression_optimization

async def main():
    results = await run_compression_optimization(
        objective="Compress for edge deployment with minimal carbon",
        carbon_budget=5.0,
        max_iterations=10,
        accuracy_threshold=0.93
    )

    print(f"Pareto 最優解數量: {results['pareto_optimal_count']}")
    print(f"最佳解決方案: {results['best_solution']}")

asyncio.run(main())
```

### 示例 2: 運行研究問題實驗

```python
import asyncio
from agentic_compression.optimization.agent_driven import run_rq2_experiment

async def main():
    # 運行 RQ2: 代理驅動優化
    results = await run_rq2_experiment(
        model="google/gemma-12b",
        accuracy_threshold=0.93,
        carbon_budget=5.0
    )

    print("碳影響分析:", results['carbon_impact_analysis'])
    print("關鍵發現:", results['key_findings'])

asyncio.run(main())
```

### 示例 3: 使用命令行

```bash
# 運行簡單優化示例
python examples/simple_optimization.py

# 運行所有實驗
python examples/run_all_experiments.py
```

---

## 🧪 運行測試

```bash
# 設置 Python 路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 運行所有測試
pytest tests/ -v

# 運行特定測試
pytest tests/test_core/test_config.py -v

# 運行帶覆蓋率的測試
pytest --cov=agentic_compression tests/
```

---

## 📁 項目結構

```
Green_AI/
├── app.py                          # Streamlit 主應用
├── requirements.txt                # 依賴列表
├── src/
│   └── agentic_compression/
│       ├── core/                   # 核心模組
│       │   ├── config.py          # 配置類
│       │   ├── metrics.py         # 評估指標
│       │   └── pareto.py          # Pareto 算法
│       ├── optimization/           # RQ 實現
│       │   ├── agent_driven.py    # RQ2
│       │   ├── dynamic_vs_static.py # RQ1
│       │   ├── weighting.py       # RQ3
│       │   └── resource_adaptation.py # RQ4
│       ├── tools/                  # LangChain 工具
│       ├── graph/                  # LangGraph 工作流
│       ├── visualization/          # 可視化
│       └── ui/                     # Streamlit UI
│           ├── components.py      # UI 組件
│           ├── visualizations.py  # 圖表
│           ├── utils.py           # 工具函數
│           └── pages/             # 4 個頁面
├── examples/                       # 示例腳本
├── tests/                          # 測試套件
└── docs/                           # 文檔
```

---

## 🔧 常見問題

### Q1: ImportError: No module named 'agentic_compression'
**解決方案:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Q2: Streamlit 無法啟動
**解決方案:**
```bash
# 確保安裝了 Streamlit
pip install streamlit

# 檢查版本
streamlit --version

# 重新運行
streamlit run app.py
```

### Q3: 缺少依賴
**解決方案:**
```bash
# 重新安裝所有依賴
pip install -r requirements.txt --upgrade
```

### Q4: 測試失敗
**解決方案:**
```bash
# 確保設置了 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 安裝測試依賴
pip install pytest pytest-asyncio pytest-cov
```

---

## 📚 下一步

1. **探索 Web UI** - 熟悉 4 個互動頁面
2. **運行示例** - 執行 `examples/` 中的腳本
3. **閱讀文檔** - 查看 `README.md` 和 `IMPLEMENTATION_SUMMARY.md`
4. **自定義實驗** - 修改配置並運行自己的實驗
5. **查看研究論文** - 了解理論背景

---

## 🆘 獲取幫助

- **文檔**: 查看 `README.md`
- **實現摘要**: 查看 `IMPLEMENTATION_SUMMARY.md`
- **代碼指南**: 查看 `CLAUDE.md`
- **問題報告**: 在 GitHub 提交 issue

---

## ✅ 驗證安裝

運行以下命令確認一切正常：

```bash
# 1. 檢查 Python 版本
python --version  # 應該是 3.10+

# 2. 檢查依賴
pip list | grep streamlit
pip list | grep plotly
pip list | grep pandas

# 3. 檢查文件結構
ls -la src/agentic_compression/ui/
ls -la src/agentic_compression/ui/pages/

# 4. 啟動 UI
streamlit run app.py
```

如果所有步驟都成功，您已準備好開始使用框架！🎉

---

**版本**: 2.0.0
**最後更新**: 2025-01-28
**狀態**: 生產就緒
