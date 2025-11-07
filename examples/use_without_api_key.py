"""
使用 Deep Agent 工具 - 無需 API Key

展示如何在沒有 API key 的情況下使用所有壓縮工具。
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_1_lora_estimation():
    """示例 1: 估算 LoRA 壓縮效果"""
    print("\n" + "=" * 80)
    print("示例 1: 估算 LoRA 壓縮效果（無需 API key）")
    print("=" * 80)

    from src.agentic_compression.agents.sub_agents.lora_sub_agent import (
        EstimateLoRAImpactTool,
        ConfigureLoRATool,
    )

    # 估算不同 rank 的 LoRA 影響
    tool = EstimateLoRAImpactTool()

    for rank in [4, 8, 16]:
        print(f"\n🔍 測試 LoRA rank={rank}:")
        result = tool._run(
            base_model="meta-llama/Llama-2-7b-hf",
            rank=rank,
        )
        result_dict = json.loads(result)
        print(f"  壓縮比: {result_dict['compression_ratio']}x")
        print(f"  可訓練參數: {result_dict['trainable_params_percent']}%")
        print(f"  推薦用途: {result_dict['recommended_use_cases'][0]}")

    # 配置 LoRA
    print(f"\n⚙️  配置 LoRA (rank=8):")
    config_tool = ConfigureLoRATool()
    result = config_tool._run(
        base_model="meta-llama/Llama-2-7b-hf",
        rank=8,
        alpha=16,
    )
    config = json.loads(result)
    print(f"  目標模塊: {config['config']['target_modules']}")
    print(f"  適配器大小: {config['estimated_adapter_size_m']} M 參數")


def example_2_distillation_planning():
    """示例 2: 規劃知識蒸餾"""
    print("\n" + "=" * 80)
    print("示例 2: 規劃知識蒸餾（無需 API key）")
    print("=" * 80)

    from src.agentic_compression.agents.sub_agents.distillation_sub_agent import (
        SetupDistillationTool,
        EstimateDistillationTool,
    )

    # 設置蒸餾
    setup_tool = SetupDistillationTool()
    result = setup_tool._run(
        teacher_model="meta-llama/Llama-2-7b-hf",
        student_scale=0.5,  # 學生模型是老師的 50%
        temperature=2.0,
        alpha=0.7,
    )
    config = json.loads(result)

    print(f"\n📚 蒸餾配置:")
    print(f"  老師: {config['teacher']['model_name']} ({config['teacher']['size_m']} M)")
    print(f"  學生: {config['student']['size_m']} M")
    print(f"  壓縮比: {config['compression_ratio']}x")
    print(f"  預期精度損失: {config['expected_accuracy_loss']}")
    print(f"  預期加速: {config['estimated_speedup']}")

    # 估算不同壓縮比
    print(f"\n🔬 比較不同壓縮比:")
    estimate_tool = EstimateDistillationTool()

    for ratio in [2.0, 3.0, 5.0]:
        result = estimate_tool._run(
            teacher_model="meta-llama/Llama-2-7b-hf",
            compression_ratio=ratio,
        )
        est = json.loads(result)
        metrics = est['estimated_metrics']
        print(f"  {ratio}x 壓縮:")
        print(f"    精度損失: {metrics['accuracy_loss_percent']}%")
        print(f"    加速: {metrics['speedup']}")


def example_3_experiment_tracking():
    """示例 3: 實驗追蹤和分析"""
    print("\n" + "=" * 80)
    print("示例 3: 實驗追蹤和分析（無需 API key）")
    print("=" * 80)

    from src.agentic_compression.agents.tracking_tool import (
        LogExperimentTool,
        QueryExperimentsTool,
        GetBestConfigTool,
    )

    # 記錄幾個實驗
    log_tool = LogExperimentTool()

    experiments = [
        {
            "config": {"technique": "quantization", "bits": 8, "method": "gptq"},
            "metrics": {"accuracy": 0.654, "latency_ms": 45.3, "memory_mb": 3421},
        },
        {
            "config": {"technique": "quantization", "bits": 4, "method": "gptq"},
            "metrics": {"accuracy": 0.612, "latency_ms": 28.1, "memory_mb": 1800},
        },
        {
            "config": {"technique": "pruning", "sparsity": 0.3, "pattern": "2:4"},
            "metrics": {"accuracy": 0.648, "latency_ms": 38.2, "memory_mb": 2900},
        },
    ]

    print("\n📝 記錄實驗到 MLflow:")
    for i, exp in enumerate(experiments, 1):
        result = log_tool._run(
            config=exp["config"],
            metrics=exp["metrics"],
            model_name="llama-2-7b",
            tags={"batch": "demo"},
        )
        print(f"  {i}. {result[:60]}...")

    # 查詢實驗
    print(f"\n🔍 查詢量化實驗:")
    query_tool = QueryExperimentsTool()
    result = query_tool._run(
        filter_string="params.technique = 'quantization'",
        max_results=5,
        order_by="metrics.accuracy DESC",
    )
    results = json.loads(result)
    for exp in results[:2]:  # 顯示前 2 個
        print(f"  Bits: {exp['params'].get('bits', 'N/A')}")
        print(f"  Accuracy: {exp['metrics'].get('accuracy', 'N/A')}")
        print(f"  Latency: {exp['metrics'].get('latency_ms', 'N/A')} ms")

    # 獲取最佳配置
    print(f"\n🏆 最佳精度配置:")
    best_tool = GetBestConfigTool()
    result = best_tool._run(
        metric="accuracy",
        higher_is_better=True,
    )
    best = json.loads(result)
    print(f"  配置: {best['best_config']}")
    print(f"  精度: {best['metric_value']}")


def example_4_workspace_management():
    """示例 4: Workspace 記憶管理"""
    print("\n" + "=" * 80)
    print("示例 4: Workspace 記憶管理（無需 API key）")
    print("=" * 80)

    from src.agentic_compression.agents.compression_deep_agent import WorkspaceManager

    workspace = WorkspaceManager("./workspace")

    # 保存實驗
    print("\n💾 保存實驗到 workspace:")
    workspace.save_experiment(
        experiment_id="exp_quant_8bit_001",
        config={"technique": "quantization", "bits": 8},
        metrics={"accuracy": 0.654, "speedup": 2.1},
    )
    print("  ✓ 實驗已保存")

    # 保存知識
    print("\n📚 保存學習到的知識:")
    workspace.save_knowledge(
        topic="quantization_best_practices",
        content="""# 量化最佳實踐

## 8-bit 量化
- 精度損失: <1%
- 加速: 2-2.5x
- 適用場景: 大多數部署

## 4-bit 量化
- 精度損失: 2-3%
- 加速: 3-4x
- 適用場景: 邊緣設備、資源受限環境
- 建議: 使用 GPTQ 或 AWQ 方法

## 注意事項
- 始終在目標數據集上評估
- 考慮校準數據的質量
- 監控離群值的影響
""",
    )
    print("  ✓ 知識已保存")

    # 讀取知識
    print("\n📖 讀取已保存的知識:")
    content = workspace.load_knowledge("quantization_best_practices")
    print(content[:200] + "...")


def example_5_comparison_report():
    """示例 5: 生成壓縮技術比較報告"""
    print("\n" + "=" * 80)
    print("示例 5: 壓縮技術比較報告（無需 API key）")
    print("=" * 80)

    from src.agentic_compression.agents.sub_agents.lora_sub_agent import (
        EstimateLoRAImpactTool,
    )
    from src.agentic_compression.agents.sub_agents.distillation_sub_agent import (
        EstimateDistillationTool,
    )

    # 模型規格
    model = "meta-llama/Llama-2-7b-hf"

    # LoRA
    lora_tool = EstimateLoRAImpactTool()
    lora_result = json.loads(lora_tool._run(base_model=model, rank=8))

    # 蒸餾
    distill_tool = EstimateDistillationTool()
    distill_result = json.loads(
        distill_tool._run(teacher_model=model, compression_ratio=2.0)
    )

    # 量化（手動估算）
    quant_8bit = {
        "compression_ratio": 4.0,
        "accuracy_loss": 0.5,
        "speedup": "2.0x",
    }

    quant_4bit = {
        "compression_ratio": 8.0,
        "accuracy_loss": 2.0,
        "speedup": "3.0x",
    }

    print(f"\n📊 壓縮技術比較 - {model}\n")
    print("技術                壓縮比    精度損失    加速        適用場景")
    print("-" * 80)

    print(
        f"{'8-bit 量化':<15} {quant_8bit['compression_ratio']:>7.1f}x  "
        f"{quant_8bit['accuracy_loss']:>8.1f}%  {quant_8bit['speedup']:>8}  "
        f"通用部署"
    )

    print(
        f"{'4-bit 量化':<15} {quant_4bit['compression_ratio']:>7.1f}x  "
        f"{quant_4bit['accuracy_loss']:>8.1f}%  {quant_4bit['speedup']:>8}  "
        f"邊緣設備"
    )

    print(
        f"{'LoRA (rank=8)':<15} {lora_result['compression_ratio']:>7.1f}x  "
        f"{'<1.0':>8}%  {'1.0x':>8}  "
        f"任務微調"
    )

    distill_metrics = distill_result["estimated_metrics"]
    print(
        f"{'蒸餾 (2x)':<15} {distill_result['compression_ratio']:>7.1f}x  "
        f"{distill_metrics['accuracy_loss_percent']:>8.1f}%  {distill_metrics['speedup']:>8}  "
        f"模型壓縮"
    )

    print("\n💡 推薦:")
    print("  • 需要高精度: 8-bit 量化")
    print("  • 任務特定優化: LoRA")
    print("  • 極致壓縮: 蒸餾 + 4-bit 量化")
    print("  • 邊緣部署: 4-bit 量化")


def main():
    """運行所有示例"""
    print("\n" + "=" * 80)
    print("🎉 Deep Agent 工具使用示例 - 無需 API Key")
    print("=" * 80)

    example_1_lora_estimation()
    example_2_distillation_planning()
    example_3_experiment_tracking()
    example_4_workspace_management()
    example_5_comparison_report()

    print("\n" + "=" * 80)
    print("✅ 所有示例完成！")
    print("\n💡 提示:")
    print("  • 這些工具都不需要 API key")
    print("  • 可以直接在你的代碼中使用")
    print("  • 查看 MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    print("  • 查看 workspace: ls -la workspace/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
