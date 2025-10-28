# 🎯 執行指令清單

**請按順序執行以下指令**

---

## ✅ 步驟 1: 安裝依賴（必須）

```bash
pip install -r requirements.txt
```

---

## ✅ 步驟 2: 啟動 Streamlit UI（必須）

```bash
streamlit run app.py
```

**預期輸出**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**瀏覽器應該自動打開，如果沒有，手動訪問:** `http://localhost:8501`

---

## ✅ 步驟 3: 驗證 UI 功能（建議）

在打開的瀏覽器中檢查：

1. **主頁**:
   - [ ] 看到綠色的大標題 "🌱 Agentic Compression Framework"
   - [ ] 看到 4 個功能卡片
   - [ ] 側邊欄有導航選項

2. **側邊欄**:
   - [ ] 看到 4 個頁面選項：
     - 🚀 Quick Optimization
     - 📊 Advanced Visualization
     - 🔬 Experiment Comparison
     - 🎯 Interactive 3D Explorer

3. **快速測試 - Quick Optimization**:
   - 點擊側邊欄 "1_Quick_Optimization"
   - 在側邊欄配置參數（使用默認值即可）
   - 點擊 "▶️ Run Optimization"
   - 等待約 10-30 秒
   - [ ] 看到優化進度
   - [ ] 看到結果顯示（摘要統計、最佳方案、圖表）

---

## ✅ 步驟 4: 運行測試（可選）

```bash
# 設置 Python 路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 運行核心測試
pytest tests/test_core/ -v

# 預期: 所有測試通過
```

---

## ✅ 步驟 5: 運行示例程序（可選）

```bash
# 設置路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 運行簡單示例
python examples/simple_optimization.py

# 預期: 顯示優化過程和結果
```

---

## 📊 驗證清單

完成後，確認以下項目：

- [ ] Streamlit UI 成功啟動
- [ ] 可以看到 4 個頁面
- [ ] 主頁顯示正常
- [ ] 側邊欄配置表單可用
- [ ] 可以運行優化實驗
- [ ] 沒有 Python import 錯誤

---

## 🐛 常見錯誤排除

### 錯誤 1: `ModuleNotFoundError: No module named 'streamlit'`
**解決:**
```bash
pip install streamlit plotly pandas
```

### 錯誤 2: `ModuleNotFoundError: No module named 'agentic_compression'`
**解決:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 錯誤 3: Streamlit 端口已被占用
**解決:**
```bash
# 使用不同端口
streamlit run app.py --server.port 8502
```

### 錯誤 4: 測試失敗
**解決:**
```bash
# 確保路徑設置正確
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 重新運行測試
pytest tests/ -v
```

---

## 📝 成功標誌

如果您看到以下內容，說明一切正常：

1. ✅ Streamlit UI 在瀏覽器中打開
2. ✅ 可以看到漂亮的綠色主頁
3. ✅ 4 個頁面都可以訪問
4. ✅ 可以配置參數並運行優化
5. ✅ 沒有紅色錯誤消息

---

## 🎉 下一步

成功後，您可以：

1. **探索所有頁面** - 熟悉每個功能
2. **運行實驗** - 測試不同的配置
3. **查看文檔** - 閱讀 `QUICKSTART.md` 和 `README.md`
4. **自定義** - 修改代碼以滿足您的需求

---

## 💡 快速提示

**最快的驗證方式:**

```bash
# 一條命令安裝並啟動
pip install streamlit plotly pandas && streamlit run app.py
```

瀏覽器打開後，點擊幾個頁面，如果都能正常顯示，就大功告成了！🎊

---

**需要幫助?** 查看 `QUICKSTART.md` 獲取詳細指南
