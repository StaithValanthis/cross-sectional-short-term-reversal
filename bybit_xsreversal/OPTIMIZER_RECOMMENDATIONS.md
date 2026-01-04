# Optimizer Recommendations for Thorough Testing

This document provides recommended configurations for running the optimizer with industry best practices.

## 🎯 Recommended: Production-Ready Thorough Test

This configuration uses **walk-forward analysis**, **composite objective function**, and **OOS validation** to ensure robust parameter selection.

### Quick Start

```bash
cd bybit_xsreversal
./scripts/optimize_thorough.sh
```

### Manual Configuration

```bash
cd bybit_xsreversal
source .venv/bin/activate

# Data window: 2 years for robust testing
export BYBIT_OPT_WINDOW_DAYS=730

# Train/test split: 70/30 with minimum 6 months OOS
export BYBIT_OPT_TRAIN_FRAC=0.7
export BYBIT_OPT_MIN_TEST_DAYS=180

# Walk-forward: 5 rolling windows (30-day steps)
export BYBIT_OPT_WALK_FORWARD=1
export BYBIT_OPT_WF_NUM_WINDOWS=5
export BYBIT_OPT_WF_WINDOW_STEP_DAYS=30

# Composite objective (better than pure Sharpe)
export BYBIT_OPT_USE_COMPOSITE=1
export BYBIT_OPT_OBJ_WEIGHT_SHARPE=0.4
export BYBIT_OPT_OBJ_WEIGHT_CALMAR=0.3
export BYBIT_OPT_OBJ_WEIGHT_CAGR=0.2
export BYBIT_OPT_OBJ_WEIGHT_TURNOVER=0.1

# Profitability gates
export BYBIT_OPT_MIN_SHARPE=0.3          # Minimum train Sharpe
export BYBIT_OPT_REQUIRE_OOS=1           # Require OOS validation
export BYBIT_OPT_MIN_OOS_SHARPE=0.2      # Minimum OOS Sharpe
export BYBIT_OPT_MIN_OOS_CAGR=0.0        # Minimum OOS CAGR

# Run optimization
bybit-xsreversal --config config/config.yaml optimize \
    --level deep \
    --method random \
    --candidates 10000 \
    --stage2-topk 400 \
    --seed 42 \
    --no-write
```

### What This Does

1. **Walk-Forward Analysis**: Tests parameters across 5 rolling windows to ensure robustness
2. **Composite Objective**: Ranks by weighted combination of Sharpe (40%), Calmar (30%), CAGR (20%), and turnover penalty (10%)
3. **OOS Validation**: Only accepts configs that are profitable out-of-sample
4. **Large Candidate Pool**: Tests 10,000 candidates in Stage1, top 400 in Stage2
5. **2-Year Window**: Uses 730 days of history for comprehensive testing

### Expected Runtime

- **Stage1**: ~2-3 hours (10,000 candidates × 5 windows = 50,000 total evaluations)
- **Stage2**: ~6-8 hours (400 candidates × 5 windows = 2,000 full backtests)
- **Total**: ~8-12 hours

---

## 🔬 Maximum Thoroughness (Research Mode)

For maximum confidence, run multiple seeds and aggregate results:

```bash
# Run 5 different seeds
for seed in 42 123 456 789 999; do
    export BYBIT_OPT_WINDOW_DAYS=730
    export BYBIT_OPT_TRAIN_FRAC=0.7
    export BYBIT_OPT_MIN_TEST_DAYS=180
    export BYBIT_OPT_WALK_FORWARD=1
    export BYBIT_OPT_WF_NUM_WINDOWS=7  # More windows
    export BYBIT_OPT_WF_WINDOW_STEP_DAYS=20  # Smaller steps
    export BYBIT_OPT_USE_COMPOSITE=1
    export BYBIT_OPT_REQUIRE_OOS=1
    export BYBIT_OPT_MIN_OOS_SHARPE=0.2
    
    bybit-xsreversal --config config/config.yaml optimize \
        --level deep \
        --method random \
        --candidates 20000 \
        --stage2-topk 500 \
        --seed $seed \
        --no-write \
        --output-dir "outputs/optimize/max-thorough-seed-${seed}"
done
```

**Expected Runtime**: ~40-60 hours total (8-12 hours per seed)

---

## ⚡ Quick Test (Validation)

For quick validation of changes (not production-ready):

```bash
export BYBIT_OPT_WINDOW_DAYS=365
export BYBIT_OPT_TRAIN_FRAC=0.7
export BYBIT_OPT_MIN_TEST_DAYS=60
# No walk-forward for speed
# No composite (use default lexicographic)

bybit-xsreversal --config config/config.yaml optimize \
    --level quick \
    --method random \
    --candidates 1000 \
    --stage2-topk 25 \
    --seed 42 \
    --no-write
```

**Expected Runtime**: ~30-60 minutes

---

## 📊 Understanding Results

### Walk-Forward Summary

After a walk-forward run, check `walkforward_summary.json`:

```json
{
  "num_windows": 5,
  "best_candidate": {...},
  "best_score": 1.234,
  "appearance_count": 4,  // Appeared in 4/5 windows (robust!)
  "candidate_rankings": [
    {
      "candidate": {...},
      "score": 1.234,
      "appearance_count": 4,
      "avg_oos_sharpe": 0.85
    }
  ]
}
```

**Key Metrics**:
- **appearance_count**: Higher is better (robustness across windows)
- **avg_oos_sharpe**: Average out-of-sample Sharpe across windows
- **score**: Composite score (weighted combination)

### OOS Metrics

Check `oos_best.json` for train vs test comparison:

```json
{
  "train": {
    "sharpe": 1.5,
    "cagr": 0.25,
    "calmar": 2.0
  },
  "test": {
    "sharpe": 0.8,
    "cagr": 0.15,
    "calmar": 1.2,
    "overfitting_warning": "Sharpe degradation: 46.7% (train=1.500 test=0.800)"
  }
}
```

**Red Flags**:
- **Overfitting warning**: Test metrics < 50% of train metrics
- **Negative OOS Sharpe**: Strategy doesn't work out-of-sample
- **Large degradation**: >50% drop from train to test

---

## 🎛️ Parameter Tuning Guide

### If No Candidates Found

**Relax Stage1 gates**:
```bash
export BYBIT_OPT_STAGE1_MAX_DD_PCT=50    # Allow up to 50% drawdown
export BYBIT_OPT_STAGE1_MAX_TURNOVER=5.0  # Allow higher turnover
```

### If Too Many Candidates Pass Stage1

**Tighten Stage1 gates**:
```bash
export BYBIT_OPT_STAGE1_MAX_DD_PCT=20    # Stricter drawdown limit
export BYBIT_OPT_STAGE1_MAX_TURNOVER=2.0  # Lower turnover limit
```

### If OOS Performance is Poor

**Increase OOS requirements**:
```bash
export BYBIT_OPT_MIN_OOS_SHARPE=0.5      # Require 0.5+ OOS Sharpe
export BYBIT_OPT_MIN_OOS_CAGR=0.1         # Require 10%+ OOS CAGR
```

### If Walk-Forward Takes Too Long

**Reduce windows**:
```bash
export BYBIT_OPT_WF_NUM_WINDOWS=3         # Fewer windows
export BYBIT_OPT_WF_WINDOW_STEP_DAYS=60   # Larger steps
```

---

## 📈 Interpreting Results

### Good Signs ✅

- **High appearance_count**: Candidate appears in 4+/5 windows
- **Positive OOS Sharpe**: >0.3 is good, >0.5 is excellent
- **Low overfitting**: Test metrics within 20% of train metrics
- **Consistent Calmar**: High Calmar ratio (>1.0) indicates good risk-adjusted returns

### Warning Signs ⚠️

- **Low appearance_count**: Only appears in 1-2 windows (not robust)
- **Negative OOS Sharpe**: Strategy fails out-of-sample
- **Overfitting detected**: Large train/test gap (>50% degradation)
- **High turnover**: >2.0 indicates excessive trading costs

### Red Flags 🚩

- **Negative OOS CAGR**: Strategy loses money out-of-sample
- **Extreme overfitting**: Test Sharpe < 30% of train Sharpe
- **Inconsistent windows**: Best candidate changes dramatically between windows

---

## 🔄 Continuous Optimization

For ongoing optimization (e.g., monthly re-optimization):

1. **Use shorter windows**: `BYBIT_OPT_WINDOW_DAYS=365` (1 year)
2. **Fewer walk-forward windows**: `BYBIT_OPT_WF_NUM_WINDOWS=3`
3. **Lower candidate count**: `--candidates 5000`
4. **Keep OOS validation**: `BYBIT_OPT_REQUIRE_OOS=1`

This balances thoroughness with runtime (~4-6 hours).

---

## 💡 Best Practices

1. **Always use walk-forward** for production configs
2. **Require OOS profitability** (`BYBIT_OPT_REQUIRE_OOS=1`)
3. **Use composite objective** for better ranking
4. **Check overfitting warnings** before deploying
5. **Run multiple seeds** for maximum confidence
6. **Review `walkforward_summary.json`** to understand robustness
7. **Test on testnet first** before live deployment

---

## 📝 Example: Full Production Run

```bash
#!/bin/bash
# Production-ready optimization with all best practices

cd bybit_xsreversal
source .venv/bin/activate

export BYBIT_OPT_WINDOW_DAYS=730
export BYBIT_OPT_TRAIN_FRAC=0.7
export BYBIT_OPT_MIN_TEST_DAYS=180
export BYBIT_OPT_WALK_FORWARD=1
export BYBIT_OPT_WF_NUM_WINDOWS=5
export BYBIT_OPT_WF_WINDOW_STEP_DAYS=30
export BYBIT_OPT_USE_COMPOSITE=1
export BYBIT_OPT_OBJ_WEIGHT_SHARPE=0.4
export BYBIT_OPT_OBJ_WEIGHT_CALMAR=0.3
export BYBIT_OPT_OBJ_WEIGHT_CAGR=0.2
export BYBIT_OPT_OBJ_WEIGHT_TURNOVER=0.1
export BYBIT_OPT_MIN_SHARPE=0.3
export BYBIT_OPT_REQUIRE_OOS=1
export BYBIT_OPT_MIN_OOS_SHARPE=0.2
export BYBIT_OPT_MIN_OOS_CAGR=0.0

bybit-xsreversal --config config/config.yaml optimize \
    --level deep \
    --method random \
    --candidates 10000 \
    --stage2-topk 400 \
    --seed 42 \
    --no-write
```

**This is the recommended configuration for production use.**

