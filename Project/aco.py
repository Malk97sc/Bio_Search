import random
import time

class AntColony:
    def __init__(self, items, max_capacity, ant_count=10, generations=100, alpha=1.0, beta=2.0, rho=0.1, Q=100):
        self.items = items                                  #lista de objetos disponibles
        self.N = len(items)                                 #cantidad de objetos distintos
        self.max_capacity = max_capacity                    #capacidad máxima de la mochila

        #parametros de la colonia
        self.ant_count = ant_count                          #numero de hormigas por generacion
        self.generations = generations                      #cantidad maxima de iteraciones
        self.alpha = alpha                                  #peso de las feromonas
        self.beta = beta                                    #peso de la heuristica (valor/peso)
        self.rho = rho                                      #tasa de evaporacion de feromonas
        self.Q = Q                                          #intensidad de la actualización de feromonas

        #resultados 
        self.bestSolution = None                            #mejor solucion encontrada
        self.bestValue = 0                                  #valor de la mejor solucion
        self.fitnessList = []                               #historial de mejores valores por generacion

        #inicializacion de feromonas/heuristica
        self.pheromone = [random.uniform(0.8, 1.2) for _ in range(self.N)]  #para evitar sesgos desde el inicio
        self.heuristic = [item.value / item.weight for item in self.items]  #valor/peso por objeto

    def evaluate(self, solution):
        totalW = 0
        totalV = 0
        for x, item in zip(solution, self.items): #acumula peso y valor de los items elegidos
            totalW += x * item.weight
            totalV += x * item.value
        return totalV, totalW, totalW <= self.max_capacity

    def construct_solution(self):
        solution = [0] * self.N
        total_weight = 0

        while total_weight < self.max_capacity:
            probabilities = []
            for i in range(self.N):
                if solution[i] >= self.items[i].quantity:  #no puede exceder la cantidad disponible
                    probabilities.append(0)
                else:
                    desirability = (self.pheromone[i] ** self.alpha) * (self.heuristic[i] ** self.beta)
                    probabilities.append(desirability) #se calcula la probabilidad ponderada

            total = sum(probabilities)
            if total == 0:
                break

            i = random.choices(range(self.N), weights=probabilities, k=1)[0]
            if total_weight + self.items[i].weight <= self.max_capacity:
                solution[i] += 1
                total_weight += self.items[i].weight
            else:
                break

        value, weight, valid = self.evaluate(solution) #se evalua la solucion construida
        return solution, value, weight, valid

    def run(self):
        start_time = time.time()

        for gen in range(self.generations): 
            all_solutions = []
            all_values = []

            for _ in range(self.ant_count): #por cada hormiga
                solution, value, weight, valid = self.construct_solution()
                if valid:
                    all_solutions.append((solution, value))
                    all_values.append(value)
                    if value > self.bestValue: #actualiza si es la mejor global
                        self.bestValue = value
                        self.bestSolution = solution

            #se actualizan las feromonas
            delta_pheromone = [0.0 for _ in range(self.N)]
            for sol, val in all_solutions:
                for i in range(self.N):
                    if sol[i] > 0:
                        delta_pheromone[i] += (self.Q * val) #se calculas las feromonas en base al valor

            for i in range(self.N):
                self.pheromone[i] = (1 - self.rho) * self.pheromone[i] + delta_pheromone[i] #evaporacion

            self.fitnessList.append(self.bestValue) #esto es solo para el registro de convergencia

        bestV = max(self.fitnessList)
        idx = next((i + 1 for i, val in enumerate(self.fitnessList) if val == bestV), self.generations)
        elapsed_time = time.time() - start_time

        return {
            "bestSolution": self.bestSolution,
            "bestValue": self.bestValue,
            "iterations": self.generations,
            "time": elapsed_time,
            "convergence": self.fitnessList,
            "convergence_index": idx
        }
