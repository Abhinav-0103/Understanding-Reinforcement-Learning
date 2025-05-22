# Multi-Armed Bandit Algorithms: Final Comparison Report

This report summarizes the empirical performance of three multi-armed bandit algorithms — **Epsilon-Greedy**, **UCB (Upper Confidence Bound)**, and **Gaussian Thompson Sampling** — evaluated over 2,500 steps and 2,000 independent runs in a 10-armed Gaussian bandit setting.

---

## 🧪 Algorithm Insights

### 🎯 Epsilon-Greedy Agent

- The average reward plateaus early and the agent eventually converges to selecting the optimal action.
- However, due to its persistent random exploration, it continues to incur unnecessary regret over time.
- **Highly sensitive to the value of epsilon (ε)**: small ε results in slow exploration; large ε leads to excessive randomness.

---

### 📈 UCB Agent

- Performs well across a range of steps.
- **Lower `c` values** (e.g., 2) tend to result in stable and efficient learning.
- **Higher `c` values** (e.g., 100) introduce instability and over-exploration.
- The agent plateaus early in terms of average reward and reliably converges to the optimal action.

---

### 🧠 Gaussian Thompson Sampling Agent

- Displays early convergence both in terms of average reward and percentage of optimal action.
- **Requires minimal hyperparameter tuning**, making it highly practical.
- Offers stable and robust performance across runs.

---

## 📊 Final Comparison Summary

### Performance Highlights:
- Both **UCB** and **Gaussian Thompson Sampling (GS)** outperform Epsilon-Greedy in terms of convergence speed and stability.
- UCB slightly edges out GS in **cumulative regret**, although the difference is minimal.
- Epsilon-Greedy accumulates significantly higher regret due to its persistent random exploration.

### Verdict:
- While **UCB** and **GS** both demonstrate strong performance, **Gaussian Thompson Sampling** is preferred due to its **lack of hyperparameter sensitivity** and consistent results.
- Therefore, **Gaussian Thompson Sampling is considered the best overall** in this setting.

---

## 🖼️ Final Comparison Plots

![Final Comparison of Average Reward and % Optimal Action](bandits/Final-Comparison - 3000 runs.png)

> _Figure: Comparison of average reward and % optimal action across Epsilon-Greedy, UCB (c=2), and Gaussian Thompson Sampling over 2500 steps and 2000 runs._

---

## 📁 Files & Reproducibility

- `bandit_env.py`: Contains Bandit environment definitions
- `agents.py`: Implements Epsilon-Greedy, UCB, and Gaussian TS agents
- `run_experiments.py`: Experiment runner and plotting utilities
- `final_comparison.png`: Visualization included above

---

## ✅ Conclusion

This benchmark confirms that **probabilistic methods like Thompson Sampling** offer both strong performance and practicality. In contrast, traditional methods like Epsilon-Greedy are useful for learning but require more careful tuning and accumulate higher regret.

---