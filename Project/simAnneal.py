import math
import random
import time

class SimAnnealKnapsack:
    def __init__(self, items, max_capacity, T=1000, alpha=0.95, stopping_T=1e-8, stopping_iter=10000):
        self.items = items #lista de items
        self.N = len(items) #number of iters
        self.max_capacity = max_capacity #capacidad maxima de la mochila

        self.T = T #temperatura inicial
        self.alpha = alpha #enfriamiento
        self.stoppingT = stopping_T #temperatura de parada
        self.stoppingIter = stopping_iter #cantidad maxima de iteraciones

        self.iteration = 0 #iteraciones actuales

        self.bestSolution = None #la mejor solucion
        self.bestValue = 0  #mejor valor
        self.fitnessList = [] #lista de fitness
    
    def initialSolution(self):
        sorted_items = sorted(enumerate(self.items), key=lambda x: x[1].value / x[1].weight + random.uniform(0, 0.1), reverse=True)

        solution = [0] * self.N
        total_weight = 0

        for i, item in sorted_items:
            max_qty = min(item.quantity, int((self.max_capacity - total_weight) / item.weight))
            if max_qty > 0:
                solution[i] = max_qty
                total_weight += max_qty * item.weight
                if total_weight >= self.max_capacity:
                    break

        value, weight, valid = self.evaluate(solution)
        return solution, value, weight, valid
    
    def evaluate(self, solution):
        totalW = 0 #peso total
        totalV = 0 #valor total 
        for x, item in zip(solution, self.items):
            totalW += x * item.weight
            totalV += x * item.value
        return totalV, totalW, totalW <= self.max_capacity

    def neighbor(self, solution):
        neigh = solution[:]
        if random.random() < 0.5:  # tweak 1‑3 unidades en un item
            for _ in range(random.randint(1, 3)):
                idx = random.randrange(self.N)
                delta = random.choice([-1, +1])
                neigh[idx] = max(0, min(self.items[idx].quantity, neigh[idx] + delta))
        else:  # swap 1 unidad entre dos items distintos
            i, j = random.sample(range(self.N), 2)
            if neigh[i] > 0:
                neigh[i] -= 1
                neigh[j] = min(self.items[j].quantity, neigh[j] + 1)
        return neigh
    
    def p_accept(self, old_val, new_val):
        if new_val > old_val: #si el valor es mejor, lo aceptamos
            return 1.0
        return math.exp((new_val - old_val) / self.T) #probabilidad basada en la temperatura

    def anneal(self):
        start_time = time.time()
        current, cur_val, _, valid = self.initialSolution()
        while not valid:
            current, cur_val, _, valid = self.initialSolution()

        self.bestSolution = current[:]
        self.bestValue = cur_val
        self.fitnessList = [cur_val]

        while self.T > self.stoppingT and self.iteration < self.stoppingIter:
            candidate = self.neighbor(current)
            cand_val, cand_weight, cand_valid = self.evaluate(candidate)

            if cand_valid and (cand_val > cur_val or random.random() < self.p_accept(cur_val, cand_val)):
                current = candidate
                cur_val = cand_val

                if cand_val > self.bestValue: #si el nuevo candidato es mejor que la mejor solucion, actualizamos
                    self.bestSolution = candidate
                    self.bestValue = cand_val

            self.fitnessList.append(cur_val)
            self.T *= self.alpha
            self.iteration += 1

        bestV = max(self.fitnessList)
        idx = next((i + 1 for i, val in enumerate(self.fitnessList) if val == bestV), self.iteration)

        elapsed_time = time.time() - start_time
        return {
            "bestSolution": self.bestSolution,
            "bestValue": self.bestValue,
            "iterations": self.iteration,
            "time": elapsed_time,
            "convergence": self.fitnessList,
            "convergence_index": idx
        }