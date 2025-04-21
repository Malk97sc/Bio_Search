import os
import mochila_data
import csv
from aco import AntColony
from simAnneal import SimAnnealKnapsack
from convergencePlotter import ConvergencePlotter

MAX_C = 20.0

def save_data(result, mochila_items, method, color, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sumW = 0
    total_qty = 0
    filas = []

    for i, qty in enumerate(result["bestSolution"]):
        if qty > 0:
            item = mochila_items[i]
            peso = round(item.weight, 4)
            valor = int(item.value)
            filas.append([method, i+1, qty, peso, valor])
            sumW += qty * item.weight
            total_qty += qty

    filas.append([method, "Total", total_qty, round(sumW, 2), int(result['bestValue'])])
    filas.append([method, "Iteraciones", "", result['iterations'], ""])
    filas.append([method, "Convergencia", "", result['convergence_index'], ""])
    filas.append([method, "Tiempo (s)", "", round(result['time'], 3), ""])

    # Guardar CSV
    csv_path = os.path.join(output_dir, "resumen_individual.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "Item", "Cantidad", "Peso", "Valor"])
        writer.writerows(filas)


    plotter = ConvergencePlotter()
    plotter.add_run(result["convergence"], label=method,
                    convergence_index=result["convergence_index"], color=color)
    plot_path = os.path.join(output_dir, "convergencia_individual.png")
    plotter.plot(title=f"Convergencia de una ejecucion individual – {method}",
                 save_path=plot_path)


def main():
    print("Ejecutando test indivual")
    path = 'data/Mochila_capacidad_maxima_20kg.xlsx'
    mochila_items = mochila_data.readItems(path)


    # SA - Configuracion original
    original = {"T":15000, "alpha":0.996, "stopping_T":1e-12, "stopping_iter":1000}
    sa = SimAnnealKnapsack(mochila_items, MAX_C, **original)
    sa_result = sa.anneal()
    save_data(sa_result, mochila_items, "SA (original)", "blue",
                       "results/test_individual/sa_original")

    # ACO - Configuracion competitiva
    competitiva = {"ant_count":12, "generations":700, "alpha":1.3, "beta":1.5, "rho":0.25, "Q":80}
    aco = AntColony(mochila_items, MAX_C, **competitiva)
    aco_result = aco.run()
    save_data(aco_result, mochila_items, "ACO (competitiva)", "red",
                       "results/test_individual/aco_competitiva")


if __name__ == '__main__':
    main()
