# Experiment Design: Large-Scale Synthetic and Real Benchmark

## 1. Objective
The primary objective of this experiment is to evaluate and compare pure statistical methods (OR), pure large language model methods (LLM), and hybrid methods (LLM-to-OR and OR-to-LLM) across a diverse range of demand patterns and lead time scenarios. We aim to demonstrate that hybrid methods offer superior robustness and performance, particularly in non-stationary and stochastic environments.

## 2. Experimental Setup

### 2.1 Agent Strategies
We will evaluate four primary agent strategies plus a theoretical upper bound:

| Strategy | Description |
| :--- | :--- |
| **OR (Baseline)** | Pure statistical base-stock policy using empirical mean/std estimation. |
| **LLM (Pure)** | Pure LLM-based decision making using the standard Vending Machine prompt. |
| **LLM-to-OR** | Hybrid: LLM analyzes history and proposes OR parameters ($\mu$, $\sigma$, $L$). |
| **OR-to-LLM** | Hybrid: OR generates recommendations/stats, which the LLM uses for final decisions. |
| **Perfect Score** | Theoretical upper bound calculated with full future demand knowledge (used for normalization). |

### 2.2 Environment
- **Simulation Platform**: TextArena `VendingMachine-v0`.
- **Review Period**: 1 (Decision made every period).
- **Inventory Parameters**: Fixed Profit = 4, Holding Cost = 1.
- **Initial State**: Starting inventory = 0.

## 3. Dataset Characteristics (1,200 Instances Total)

The benchmark is split equally between synthetic and real-world data across three lead time settings.

### 3.1 Data Categories
| Category | Settings | Count | Total |
| :--- | :--- | :--- | :--- |
| **Synthetic** | 10 Patterns × 4 Variants × 5 Realizations | 200 per LT | 600 |
| **Real (H&M)** | 200 Articles | 200 per LT | 600 |

### 3.2 Lead Time Variations (LT)
Every instance is tested under three conditions:
1. **LT = 0**: Immediate delivery.
2. **LT = 4**: Constant 4-period lag.
3. **LT = Stochastic**: Lead times sampled from $\{1, 2, inf\}$ (equally likely).

## 4. Evaluation Metrics

To allow comparison across different scales of demand, we use **Normalized Reward (NR)** as the primary metric:

\[ NR_{agent} = \frac{Reward_{agent}}{Reward_{perfect\_score}} \times 100\% \]

- **Raw Reward**: Total profit minus total holding costs over 50 periods (test set).
- **Normalized Reward**: Expressed as a percentage of the perfect score for that specific trajectory.

## 5. Execution Strategy

1. **Deterministic Seeding**: LLM agents will use `temperature=0` to ensure reproducibility.
2. **Train/Test Split**:
   - Each instance provides a `train.csv` (5 historical periods) for initial observation.
   - Evaluation is performed on the `test.csv` (50 periods).
3. **Parallel Execution**: Due to the volume (1,200 instances × 4 agents = 4,800 runs), simulations will be batched by lead time and pattern.

## 6. Output Tables

The final results will be summarized into two high-level tables for the paper.

### Table 1: Global Performance (Avg % of Perfect Score)
Groups results by data source and lead time difficulty.

| Data Source | Lead Time 0 | Lead Time 4 | LT Stochastic | **Overall** |
| :--- | :---: | :---: | :---: | :---: |
| **Synthetic (600)** | | | | |
| - OR | | | | |
| - LLM | | | | |
| - LLM-to-OR | | | | |
| - OR-to-LLM | | | | |
| **Real (600)** | | | | |
| - OR | | | | |
| - LLM | | | | |
| - LLM-to-OR | | | | |
| - OR-to-LLM | | | | |

### Table 2: Synthetic Pattern Deep-Dive (Avg % of Perfect Score)
Aggregated across all lead times for each of the 10 patterns.

| Pattern | OR | LLM | LLM-to-OR | OR-to-LLM | Winner |
| :--- | :---: | :---: | :---: | :---: | :--- |
| p01: Stationary | | | | | |
| p02: Mean Increase| | | | | |
| p03: Mean Decrease| | | | | |
| p04: Inc. Trend | | | | | |
| p05: Dec. Trend | | | | | |
| p06: Var. Change | | | | | |
| p07: Seasonal | | | | | |
| p08: Multi-CP | | | | | |
| p09: Temp Spike | | | | | |
| p10: AR(1) | | | | | |
