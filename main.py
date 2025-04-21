import os
import random
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from mochila_data import readItems, MAX_C
from simAnneal import SimAnnealKnapsack
from aco import AntColony

def compare_hyperparams(path, sa_configs, aco_configs, config_names, iters=10):
    base_dir = os.path.join("results", "comparativa_hiperparametros")
    os.makedirs(base_dir, exist_ok=True)

    summary = []
    raw_results = {}

    for name in config_names:
        items = readItems(path)
        sa_results, sa_times = [], []
        aco_results, aco_times = [], []

        raw_results[name] = {'sa': sa_results, 'aco': aco_results}

        for _ in range(iters):
            #SA
            sa = SimAnnealKnapsack(items, MAX_C, **sa_configs[name])
            res_sa = sa.anneal()
            sa_results.append(res_sa["bestValue"])
            sa_times.append(res_sa["time"])

            #ACO
            aco = AntColony(items, MAX_C, **aco_configs[name])
            res_aco = aco.run()
            aco_results.append(res_aco["bestValue"])
            aco_times.append(res_aco["time"])

        #calculamos cada una de las metricas
        metrics = {
            "config": name,
            "sa_min": np.min(sa_results),
            "sa_max": np.max(sa_results),
            "sa_mean": np.mean(sa_results),
            "sa_var": np.var(sa_results),
            "sa_time": np.mean(sa_times),
            "aco_min": np.min(aco_results),
            "aco_max": np.max(aco_results),
            "aco_mean": np.mean(aco_results),
            "aco_var": np.var(aco_results),
            "aco_time": np.mean(aco_times)
        }
        summary.append(metrics)

        #se guarda la distribucion de los resultados
        plt.figure(figsize=(10, 4))
        bins = 20
        plt.hist(sa_results, bins=bins, alpha=0.6, label='SA', edgecolor='black')
        plt.hist(aco_results, bins=bins, alpha=0.6, label='ACO', edgecolor='black')
        plt.title(f'Distribucion de resultados - {name}')
        plt.xlabel('Valor Total')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'hist_{name}.png'))
        plt.close()

    #guardamos todos los datos en en csv
    csv_path = os.path.join(base_dir, 'comparativa_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    configs = [r['config'] for r in summary]
    sa_means = [r['sa_mean'] for r in summary]
    aco_means = [r['aco_mean'] for r in summary]
    sa_times = [r['sa_time'] for r in summary]
    aco_times = [r['aco_time'] for r in summary]
    x = np.arange(len(configs))
    width = 0.35

    #media de valores
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, sa_means, width, label='SA Mean')
    plt.bar(x + width/2, aco_means, width, label='ACO Mean')
    plt.xticks(x, configs)
    plt.ylabel('Valor Medio')
    plt.title('Comparacion de Valores Medios por Configuracion')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'mean_comparison.png'))
    plt.close()

    #tiempos
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, sa_times, width, label='SA Time (s)')
    plt.bar(x + width/2, aco_times, width, label='ACO Time (s)')
    plt.xticks(x, configs)
    plt.ylabel('Tiempo Promedio (s)')
    plt.title('Comparacion de Tiempos Promedio por Configuracion')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'time_comparison.png'))
    plt.close()

    #calculamos la mejor configuracion segun la media
    best_sa = max(summary, key=lambda r: r['sa_mean'])['config']
    best_aco = max(summary, key=lambda r: r['aco_mean'])['config']

    print(f"Mejor configuracion SA: {best_sa} con media {next(r['sa_mean'] for r in summary if r['config']==best_sa):.2f}")
    print(f"Mejor configuracion ACO: {best_aco} con media {next(r['aco_mean'] for r in summary if r['config']==best_aco):.2f}")

    #mejor configuraciono
    sa_best = raw_results[best_sa]['sa']
    aco_best = raw_results[best_sa]['aco']
    plt.figure(figsize=(10, 4))
    plt.hist(sa_best, bins=20, alpha=0.6, label=f'SA ({best_sa})', edgecolor='black')
    plt.hist(aco_best, bins=20, alpha=0.6, label=f'ACO ({best_sa})', edgecolor='black')
    plt.title(f'Distribucion Mejor Configuracion - {best_sa}')
    plt.xlabel('Valor Total')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'hist_mejor_{best_sa}.png'))
    plt.close()

def plot_histogram_results(sa_results, aco_results, save_path, config_name):
    sa_values = np.array(sa_results)
    aco_values = np.array(aco_results)

    if np.all(aco_values == aco_values[0]):
        aco_values = np.append(aco_values, aco_values[0] + 1)
    if np.all(sa_values == sa_values[0]):
        sa_values = np.append(sa_values, sa_values[0] + 1)

    min_val = int(min(sa_values.min(), aco_values.min()) - 100)
    max_val = int(max(sa_values.max(), aco_values.max()) + 100)
    bins = np.linspace(min_val, max_val, 20)

    plt.figure(figsize=(12, 5))
    plt.hist(sa_values, bins=bins, color='blue', alpha=0.6, label='SA', edgecolor='black')
    plt.hist(aco_values, bins=bins, color='red', alpha=0.6, label='ACO', edgecolor='black')
    plt.xlabel('Valor Total')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribucion de Resultados - Configuracion {config_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_best_result(sa_results, aco_results, iters, save_path, config_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, iters + 1), sa_results, label="SA Final Value", marker='o', color='blue')
    plt.plot(range(1, iters + 1), aco_results, label="ACO Final Value", marker='o', color='red')
    plt.xlabel("Ejecucion")
    plt.ylabel("Valor Total")
    plt.title(f"Mejor Valor Total por Ejecucion - Configuracion {config_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_convergence(avg_sa_conv, avg_aco_conv, save_path, config_name):
    plt.figure(figsize=(10, 5))
    plt.plot(avg_sa_conv, label='SA Convergencia Promedio', color= 'blue')
    plt.plot(avg_aco_conv, label='ACO Convergencia Promedio', color = 'red')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor Total')
    plt.title(f'Convergencia Promedio - Configuracion {config_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_statistics(sa_results, sa_times, aco_results, aco_times,
                    sa_iters, aco_iters, save_path):
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "SA", "ACO"])
        writer.writerow(["Min Valor", np.min(sa_results), np.min(aco_results)])
        writer.writerow(["Max Valor", np.max(sa_results), np.max(aco_results)])
        writer.writerow(["Media Valor", np.mean(sa_results), np.mean(aco_results)])
        writer.writerow(["Varianza Valor", np.var(sa_results), np.var(aco_results)])
        writer.writerow(["Tiempo Medio (s)", np.mean(sa_times), np.mean(aco_times)])
        writer.writerow(["Varianza Tiempo", np.var(sa_times), np.var(aco_times)])
        writer.writerow(["Iter. Convergencia Media", np.mean(sa_iters), np.mean(aco_iters)])
        writer.writerow(["Iter. Convergencia Varianza", np.var(sa_iters), np.var(aco_iters)])
    
def save_comparative_summary(best_sa_res, sa_results, sa_times,
                              best_aco_res, aco_results, aco_times,
                              save_path):
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metrica", "Simulated Annealing", "Ant Colony Optimization"])
        writer.writerow(["Mejor Valor", best_sa_res["bestValue"], best_aco_res["bestValue"]])
        writer.writerow(["Tiempo Promedio (s)", np.mean(sa_times), np.mean(aco_times)])
        writer.writerow(["Varianza", np.var(sa_results), np.var(aco_results)])
        writer.writerow(["Iteraciones Promedio", np.mean([res["iterations"] for res in [best_sa_res]]),
                         np.mean([res["iterations"] for res in [best_aco_res]])])
        writer.writerow(["Iteración de Convergencia", best_sa_res["convergence_index"], best_aco_res["convergence_index"]])

def print_solution_details(method_name, items, result):
    total_weight = 0
    print(f"\n[{method_name}] Best Solution:")
    for i, qty in enumerate(result["bestSolution"]):
        if qty > 0:
            item = items[i]
            print(f" - {qty}x Item {i+1} (weight: {item.weight:.4f}kg, value: {item.value:.0f})")
            total_weight += qty * item.weight
    print(f"Total Value: ${result['bestValue']:.0f}")
    print(f"Total Weight: {total_weight:.2f} kg")
    print(f"Iterations: {result['iterations']}")
    print(f"Execution Time: {result['time']:.3f}s")
    print(f"Converged at Iteration: {result['convergence_index']}")

def save_best_solutions(items, best_sa_res, best_aco_res, save_path):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metodo", "Item", "Cantidad", "Peso", "Valor"])
        # SA
        sol_sa = best_sa_res["bestSolution"]
        total_weight_sa = 0
        for i, qty in enumerate(sol_sa):
            if qty > 0:
                item = items[i]
                writer.writerow(["SA", i+1, qty, f"{item.weight:.2f}", f"{item.value:.0f}"])
                total_weight_sa += qty * item.weight
        writer.writerow(["SA", "Total", sum(sol_sa), f"{total_weight_sa:.2f}", f"{best_sa_res['bestValue']:.0f}"])
        writer.writerow(["SA", "Iter.Conv.", best_sa_res.get("convergence_index", "N/A"), "", ""])
        writer.writerow(["SA", "Tiempo(s)", f"{best_sa_res['time']:.3f}", "", ""])
        # ACO
        sol_aco = best_aco_res["bestSolution"]
        total_weight_aco = 0
        for i, qty in enumerate(sol_aco):
            if qty > 0:
                item = items[i]
                writer.writerow(["ACO", i+1, qty, f"{item.weight:.2f}", f"{item.value:.0f}"])
                total_weight_aco += qty * item.weight
        writer.writerow(["ACO", "Total", sum(sol_aco), f"{total_weight_aco:.2f}", f"{best_aco_res['bestValue']:.0f}"])
        writer.writerow(["ACO", "Iter.Conv.", best_aco_res.get("convergence_index", "N/A"), "", ""])
        writer.writerow(["ACO", "Tiempo(s)", f"{best_aco_res['time']:.3f}", "", ""])

def run_knapsack_analysis(path, aco_config, sa_config, iters=30, config_name="default"):
    items = readItems(path)
    results_dir = os.path.join("results", f"{config_name}_SA_vs_ACO")
    os.makedirs(results_dir, exist_ok=True)

    sa_results, sa_times, sa_conv_series, sa_conv_iters = [], [], [], []
    aco_results, aco_times, aco_conv_series, aco_conv_iters = [], [], [], []
    best_sa_val, best_aco_val = -np.inf, -np.inf
    best_sa_res, best_aco_res = None, None

    for i in range(iters):
        # SA
        sa = SimAnnealKnapsack(items, MAX_C, **sa_config)
        res_sa = sa.anneal()
        sa_results.append(res_sa["bestValue"])
        sa_times.append(res_sa["time"])
        sa_conv_series.append(res_sa.get("convergence", []))
        sa_conv_iters.append(res_sa.get("convergence_index", np.nan))
        if res_sa["bestValue"] > best_sa_val:
            best_sa_val, best_sa_res = res_sa["bestValue"], res_sa

        # ACO
        aco = AntColony(items, MAX_C, **aco_config)
        res_aco = aco.run()
        aco_results.append(res_aco["bestValue"])
        aco_times.append(res_aco["time"])
        aco_conv_series.append(res_aco.get("convergence", []))
        aco_conv_iters.append(res_aco.get("convergence_index", np.nan))
        if res_aco["bestValue"] > best_aco_val:
            best_aco_val, best_aco_res = res_aco["bestValue"], res_aco


    print("\n[Mejor Solucion Global - SA y ACO]")
    print_solution_details("Mejor SA", items, best_sa_res)
    print_solution_details("Mejor ACO", items, best_aco_res)

    save_best_solutions(items, best_sa_res, best_aco_res,
                        os.path.join(results_dir, f"mejor_solucion_{config_name}.csv"))

    print(f"\nResumen Estadistico de las {iters} ejecuciones:")
    print(f"SA: Min={np.min(sa_results)}, Max={np.max(sa_results)}, Media={np.mean(sa_results):.2f}, Var={np.var(sa_results):.2f}, Tiempo Medio={np.mean(sa_times):.4f}s, Iter.Med={np.mean(sa_conv_iters):.1f}, Iter.Var={np.var(sa_conv_iters):.1f}")
    print(f"ACO: Min={np.min(aco_results)}, Max={np.max(aco_results)}, Media={np.mean(aco_results):.2f}, Var={np.var(aco_results):.2f}, Tiempo Medio={np.mean(aco_times):.4f}s, Iter.Med={np.mean(aco_conv_iters):.1f}, Iter.Var={np.var(aco_conv_iters):.1f}")

    #guardamos las estadisticas
    save_statistics(
        sa_results, sa_times, aco_results, aco_times,
        sa_conv_iters, aco_conv_iters,
        os.path.join(results_dir, f"resumen_estadistico_{config_name}.csv")
    )

    #guardamos las graficas de los mejores resultados
    plot_best_result(sa_results, aco_results, iters,
                     os.path.join(results_dir, "mejores_valores.png"),
                     config_name)
    
    #convergencia promedio
    min_len = min(len(run) for run in sa_conv_series + aco_conv_series)
    sa_avg_conv = np.mean([run[:min_len] for run in sa_conv_series], axis=0)
    aco_avg_conv = np.mean([run[:min_len] for run in aco_conv_series], axis=0)
    plot_convergence(sa_avg_conv, aco_avg_conv,
                     os.path.join(results_dir, "convergencia.png"),
                     config_name)
    plot_histogram_results(sa_results, aco_results,
                           os.path.join(results_dir, "histograma.png"), config_name)

    #Iteraciones hasta converger
    plt.figure(figsize=(6, 4))
    methods = ['SA', 'ACO']
    iter_means = [np.mean(sa_conv_iters), np.mean(aco_conv_iters)]
    plt.bar(methods, iter_means)
    plt.xlabel('Metodo')
    plt.ylabel('Iteraciones Promedio de Convergencia')
    plt.title(f'Iteraciones promedio hasta converger - {config_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'iters_convergencia.png'))
    plt.close()

def main():
    print("\nResolviendo el problema de la mochila con metodos bioinspirados")
    path = "data/Mochila_capacidad_maxima_20kg.xlsx"

    #posibles configuraciones para SA
    sa_configs = {
        "original": {"T":15000, "alpha":0.996, "stopping_T":1e-12, "stopping_iter":1000},
        "exploratoria": {"T":12000, "alpha":0.985, "stopping_T":1e-7, "stopping_iter":1000},
        "competitiva": {"T":13000, "alpha":0.991, "stopping_T":1e-11, "stopping_iter":700},
        "debil": {"T":8000, "alpha": 0.899, "stopping_T":1e-4, "stopping_iter":300}
    }

    #posibles configuraciones para ACO
    aco_configs = {
        "original": {"ant_count":10, "generations":1000, "alpha":1.0, "beta":2.0, "rho":0.1, "Q":100},
        "exploratoria": {"ant_count":20, "generations":1000, "alpha":1.1, "beta":1.0, "rho":0.3, "Q":120},
        "competitiva": {"ant_count":12, "generations":700, "alpha":1.3, "beta":1.5, "rho":0.25, "Q":80},
        "debil": {"ant_count":8, "generations":300, "alpha":0.7, "beta":0.8, "rho":0.4, "Q":50}
    }

    configs_to_compare = ['original', 'competitiva']
    select = "debil"    

    #compare_hyperparams(path, sa_configs, aco_configs, configs_to_compare, iters=50)
    run_knapsack_analysis(path, aco_configs[select], sa_configs[select], iters=50, config_name=select)

if __name__ == "__main__":
    main()