#!/bin/bash

# Docker 使用示例腳本
# 展示如何使用 Docker 運行 LLM 驅動的優化系統

set -e

echo "=== LLM 驅動優化系統 Docker 示例 ==="
echo

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[步驟]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[完成]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[信息]${NC} $1"
}

# 步驟 1: 檢查 Docker
print_step "檢查 Docker 環境..."

if ! command -v docker &> /dev/null; then
    echo "錯誤: Docker 未安裝。請先安裝 Docker。"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "錯誤: Docker 未運行。請啟動 Docker。"
    exit 1
fi

print_success "Docker 環境正常"

# 步驟 2: 檢查 API 密鑰
print_step "檢查 LLM API 密鑰..."

if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" && -z "$GOOGLE_API_KEY" ]]; then
    print_info "未檢測到 API 密鑰，將在模擬模式下運行"
    print_info "要使用真實 LLM，請設置以下環境變數之一："
    echo "  export OPENAI_API_KEY='your-openai-key'"
    echo "  export ANTHROPIC_API_KEY='your-anthropic-key'"
    echo "  export GOOGLE_API_KEY='your-google-key'"
    echo
    read -p "是否繼續使用模擬模式？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "請設置 API 密鑰後重新運行。"
        exit 1
    fi
else
    print_success "檢測到 LLM API 密鑰"
fi

# 步驟 3: 構建 Docker 映像
print_step "構建 LLM 優化 Docker 映像..."

if docker image inspect llm-compressor:latest >/dev/null 2>&1; then
    echo "發現現有映像。"
    read -p "是否重新構建？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./run_docker.sh build
    fi
else
    ./run_docker.sh build
fi

print_success "Docker 映像準備就緒"

# 步驟 4: 運行系統測試
print_step "運行 LLM 系統測試..."

echo "這將驗證所有 LLM agents 是否正確安裝..."
./run_docker.sh test

print_success "系統測試完成"

# 步驟 5: 運行基線測試
print_step "運行基線性能測量..."

echo "這將測量模型的基線性能..."
./run_docker.sh baseline

print_success "基線測量完成"

# 步驟 6: 運行保守優化
print_step "運行保守優化策略..."

echo "這將使用 LLM agents 進行保守的模型優化..."
./run_docker.sh conservative

print_success "保守優化完成"

# 步驟 7: 檢查結果
print_step "檢查優化結果..."

echo "結果文件："
find reports/ -name "*.json" -type f | head -5

if [ -f "reports/conservative/llm_optimization_results.json" ]; then
    echo
    echo "優化摘要："
    python3 -c "
import json
try:
    with open('reports/conservative/llm_optimization_results.json', 'r') as f:
        data = json.load(f)
    print(f'總配方數: {data[\"total_recipes\"]}')
    print(f'成功配方數: {data[\"successful_recipes\"]}')
    if data['results']:
        for result in data['results'][:3]:
            if result.get('success'):
                recipe_id = result.get('recipe_id', 'unknown')
                composite_score = result.get('metrics', {}).get('composite_score', 0)
                print(f'配方 {recipe_id}: 綜合評分 {composite_score:.3f}')
except Exception as e:
    print(f'無法解析結果: {e}')
"
fi

print_success "結果檢查完成"

# 完成
echo
echo "=== Docker 示例完成！ ==="
echo
echo "您已成功運行了 LLM 驅動的模型優化系統！"
echo
echo "下一步："
echo "1. 查看 reports/ 目錄中的詳細結果"
echo "2. 嘗試其他優化策略："
echo "   ./run_docker.sh aggressive     # 激進優化"
echo "   ./run_docker.sh llm-planned    # LLM 規劃的組合"
echo "3. 使用互動式 shell 探索："
echo "   ./run_docker.sh shell"
echo "4. 設置真實 API 密鑰以獲得更好的結果"
echo
echo "如需幫助，請查看 DOCKER_QUICKSTART.md"