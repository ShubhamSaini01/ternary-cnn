#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Benchmark all inference engine variants side-by-side.
#
# Engines:
#   1. INT8 dpbusd          (engine.h)           — baseline
#   2. 4-bit sign+add       (engine_4bit.h)      — exp 18
#   3. 4-bit nibble dpbusd  (engine_4bit_dpbusd.h) — exp 19
#
# Usage:
#   ./benchmarks/benchmark_all_engines.sh [threads] [warmup] [runs]
#
# Defaults: 1 thread, 50 warmup, 200 runs
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

THREADS=${1:-1}
WARMUP=${2:-50}
RUNS=${3:-200}

BIN_DIR=inference/build
MODEL_INT8=export/ternary_resnet18.bin
SCALES_INT8=export/static_scales.json
MODEL_4BIT=export/ternary_resnet18_4bit.bin
SCALES_4BIT=export/static_scales_4bit.json
TEST_DATA=export/cifar10_test.bin

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════"
echo " Ternary Engine Benchmark Suite"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Threads: $THREADS   Warmup: $WARMUP   Runs: $RUNS"
echo ""

# ─── Step 1: Build all engines ──────────────────────────────
echo -e "${BLUE}[1/4] Building all engines...${NC}"
cd inference
make all 2>&1 | tail -1
make 4bit 2>&1 | tail -1
make 4bit-dpbusd 2>&1 | tail -1
cd ..
echo "  Built: ternary_infer, ternary_infer_4bit, ternary_infer_4bit_dpbusd"
echo ""

# ─── Step 2: Check prerequisites ────────────────────────────
echo -e "${BLUE}[2/4] Checking model files...${NC}"
MISSING=0
for f in "$MODEL_INT8" "$SCALES_INT8" "$MODEL_4BIT" "$SCALES_4BIT" "$TEST_DATA"; do
    if [ ! -f "$f" ]; then
        echo -e "  ${RED}MISSING: $f${NC}"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  Missing files. Run these first:"
    echo "    python export/calibrate_and_export.py       # INT8 model"
    echo "    python export/calibrate_export_4bit.py      # 4-bit model"
    echo "    python export/export_test_data.py           # test data"
    exit 1
fi
echo "  All model files present."
echo ""

# ─── Step 3: Run benchmarks ─────────────────────────────────
echo -e "${BLUE}[3/4] Running benchmarks (OMP_NUM_THREADS=$THREADS)...${NC}"
echo ""

TMPDIR=$(mktemp -d)

# Engine 1: INT8 dpbusd (baseline)
echo -e "${GREEN}  [Engine 1] INT8 dpbusd (engine.h)${NC}"
OMP_NUM_THREADS=$THREADS $BIN_DIR/ternary_infer \
    "$MODEL_INT8" "$SCALES_INT8" "$TEST_DATA" $WARMUP $RUNS \
    2>&1 | tee "$TMPDIR/int8.txt" | grep -E "Overall|Median|Mean|Min|P95"
echo ""

# Engine 2: 4-bit sign+add
echo -e "${YELLOW}  [Engine 2] 4-bit sign+add (engine_4bit.h)${NC}"
OMP_NUM_THREADS=$THREADS $BIN_DIR/ternary_infer_4bit \
    "$MODEL_4BIT" "$SCALES_4BIT" "$TEST_DATA" $WARMUP $RUNS \
    2>&1 | tee "$TMPDIR/4bit.txt" | grep -E "Overall|Median|Mean|Min|P95"
echo ""

# Engine 3: 4-bit nibble-packed dpbusd
echo -e "${YELLOW}  [Engine 3] U4 nibble-packed dpbusd (engine_4bit_dpbusd.h)${NC}"
OMP_NUM_THREADS=$THREADS $BIN_DIR/ternary_infer_4bit_dpbusd \
    "$MODEL_4BIT" "$SCALES_4BIT" "$TEST_DATA" $WARMUP $RUNS \
    2>&1 | tee "$TMPDIR/4bit_dpbusd.txt" | grep -E "Overall|Median|Mean|Min|P95"
echo ""

# ─── Step 4: Summary table ──────────────────────────────────
echo -e "${BLUE}[4/4] Summary${NC}"
echo ""

# Parse results
parse_result() {
    local file=$1
    ACC=$(grep "Overall:" "$file" | sed 's/.*= //')
    MED=$(grep "Median:" "$file" | awk '{print $2}')
    P95=$(grep "P95:" "$file" | awk '{print $2}')
    MIN=$(grep "Min:" "$file" | awk '{print $2}')
    echo "$ACC $MED $P95 $MIN"
}

R1=$(parse_result "$TMPDIR/int8.txt")
R2=$(parse_result "$TMPDIR/4bit.txt")
R3=$(parse_result "$TMPDIR/4bit_dpbusd.txt")

ACC1=$(echo $R1 | awk '{print $1}')
MED1=$(echo $R1 | awk '{print $2}')
P951=$(echo $R1 | awk '{print $3}')
MIN1=$(echo $R1 | awk '{print $4}')

ACC2=$(echo $R2 | awk '{print $1}')
MED2=$(echo $R2 | awk '{print $2}')
P952=$(echo $R2 | awk '{print $3}')
MIN2=$(echo $R2 | awk '{print $4}')

ACC3=$(echo $R3 | awk '{print $1}')
MED3=$(echo $R3 | awk '{print $2}')
P953=$(echo $R3 | awk '{print $3}')
MIN3=$(echo $R3 | awk '{print $4}')

echo "═══════════════════════════════════════════════════════════════════════════"
printf "%-30s %10s %10s %10s %10s\n" "Engine" "Accuracy" "Median" "P95" "Min"
echo "═══════════════════════════════════════════════════════════════════════════"
printf "%-30s %10s %12s %12s %12s\n" "INT8 dpbusd (baseline)" "$ACC1" "$MED1" "$P951" "$MIN1"
printf "%-30s %10s %12s %12s %12s\n" "4-bit sign+add" "$ACC2" "$MED2" "$P952" "$MIN2"
printf "%-30s %10s %12s %12s %12s\n" "U4 nibble-packed dpbusd" "$ACC3" "$MED3" "$P953" "$MIN3"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Compute ratios
if command -v python3 &>/dev/null; then
    python3 -c "
m1, m2, m3 = $MED1, $MED2, $MED3
print(f'  sign+add vs baseline:        {m2/m1:.1f}x slower')
print(f'  nibble dpbusd vs baseline:   {m3/m1:.1f}x slower')
print()
print(f'  Threads: $THREADS')
print(f'  Runs: $RUNS (warmup: $WARMUP)')
"
fi

echo ""
echo "Full logs saved to: $TMPDIR/"
echo "  int8.txt, 4bit.txt, 4bit_dpbusd.txt"
