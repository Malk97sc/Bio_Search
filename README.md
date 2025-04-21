# Knapsack Problem – Bioinspired Methods

This project solves the classic **0/1 Knapsack Problem** using two bioinspired algorithms:  
**Simulated Annealing (SA)** and **Ant Colony Optimization (ACO)**.  
It compares their performance across different configurations in terms of solution quality, convergence behavior, and execution time.

## 🧪 How to Run

To run the main experiment using a specific configuration:

```python
# Inside main.py
select = "original"  # or "competitiva", "exploratoria", "debil"
run_knapsack_analysis(path, aco_configs[select], sa_configs[select], iters=50, config_name=select)
```

This will:
- Run both **SA and ACO** 50 times using the selected configuration.
- Save all results and plots inside: `results/<config>_SA_vs_ACO/`.

---

## Compare Multiple Configurations

To compare two or more configurations at once and generate a summary:

```python
configs_to_compare = ["original", "competitiva"]
compare_hyperparams(path, sa_configs, aco_configs, configs_to_compare, iters=50)
```

Results will be saved in: `results/comparativa_hiperparametros/`.
