import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Tuple, List
from .algorithm import Algorithm
from .constraints_functions import ConstriantsFunctionsHandler
from utils.constants import SIZE_POPULATION, GENERATIONS
from tqdm import tqdm
from .mutation_strategy import MutationStrategies
from utils.interactive_plot import setup_plot, update_plot


class Differential_Evolution(Algorithm):

    def __init__(
        self,
        objective_function: Callable,
        constraints_functions: Callable,
        bounds_constraints: Callable,
        bounds: Tuple[List, List] = ([], []),
        g_functions: List[Callable] = [],
        h_functions: List[Callable] = [],
        F: float = 0.7,
        CR: float = 0.9,
        centroid_method: bool = False,
        beta_method: bool = False,
        evolutionary_method: bool = False,
        resrand_method: bool = False,
        ADS:bool = False,
        interactive: bool = False,
    ):
        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions

        self.centroid_method = centroid_method
        self.beta_method = beta_method
        self.evolutionary_method = evolutionary_method
        self.resrand_method = resrand_method
        self.ADS = ADS
        self.interactive = interactive

        self.population = self.generate(self.upper, self.lower)
        self.NP, self.D = self.population.shape
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        self.bounds_constraints = bounds_constraints
        self._compute_fitness_and_violations_()

        self._get_gbest_pobulation_zero_()

        self.mutation_strategies = MutationStrategies(
            self.population, self.F, self.objective_function
        )

        if self.interactive:
            (
                self.scatter_plot_factible,
                self.scatter_plot_infactible,
                self.scatter_plot_gbest,
                self.ax,
            ) = setup_plot((self.lower, self.upper))
            
    def _compute_fitness_and_violations_(self):
        for index, individual in enumerate(self.population):
            fitness = self.objective_function(individual)
            self.fitness[index] = fitness

            total_de_violaciones = ConstriantsFunctionsHandler.sum_of_violations(
                self.g_functions, self.h_functions, individual
            )
            self.violations[index] = total_de_violaciones
            
    def _mutation_operator_(self, idx, generation=0):

        samples = np.random.choice(SIZE_POPULATION, 5, replace=False)

        if self.resrand_method:
            return self.res_and_rand(self.population, self.F, (self.lower, self.upper), idx)
        else:
            return self.mutation_strategies._rand1(samples)

    def res_and_rand(self, X, F, bounds, i, max_resamples=3):
        D = X.shape[1]  # Dimensionalidad del problema
        NP = X.shape[0]  # Tamaño de la población
        lower, upper = bounds  # Desempaquetar los límites

        no_res = 0
        valid = False
        while no_res < max_resamples * D and not valid:
            # Seleccionar tres índices aleatorios r1, r2, r3 que sean diferentes entre sí y diferentes de i
            indices = np.arange(NP)
            indices = np.delete(indices, i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            
            # Generar el vector mutante
            V = X[r1] + F * (X[r2] - X[r3])
            
            # Validar si el vector mutante está dentro de los límites
            valid = np.all((V >= lower) & (V <= upper))
            
            no_res += 1
        
        if not valid:
            # Reparar el vector generando componentes aleatorias dentro de los límites
            V = np.clip(V, lower, upper)
        
        return V


    def _crossover_operator_(self, target, mutant):
        dimensions = len(target)
        trial = np.copy(target)
        j_rand = np.random.randint(dimensions)

        prob_crossover = np.random.rand(dimensions) < self.CR

        trial[prob_crossover | (np.arange(dimensions) == j_rand)] = mutant[
            prob_crossover | (np.arange(dimensions) == j_rand)
        ]

        return trial

    def _selection_operator_(self, idx, trial):
        trial_fitness = self.objective_function(trial)
        trial_violations = ConstriantsFunctionsHandler.sum_of_violations(
            self.g_functions, self.h_functions, trial
        )

        current_fitness = self.fitness[idx]
        current_violations = self.violations[idx]

        if not self.constraints_functions(
            current_fitness, current_violations, trial_fitness, trial_violations
        ):
            self.fitness[idx] = trial_fitness
            self.violations[idx] = trial_violations
            self.population[idx] = trial

    def _get_gbest_pobulation_zero_(self):
        self.position_initial = 0

        self.gbest_fitness = self.fitness[self.position_initial]
        self.gbest_violation = self.violations[self.position_initial]
        self.gbest_individual = self.population[self.position_initial]

        self.update_position_gbest_population()

    def update_position_gbest_population(self):
        for idx in range(SIZE_POPULATION):
            current_fitness = self.fitness[idx]
            current_violation = self.violations[idx]

            if not self.constraints_functions(
                self.gbest_fitness,
                self.gbest_violation,
                current_fitness,
                current_violation,
            ):
                self.gbest_fitness = current_fitness
                self.gbest_violation = current_violation
                self.gbest_individual = self.population[idx]

                self.best_fitness = self.gbest_fitness
                self.best_violations = self.gbest_violation

                self.best_fitness = self.gbest_fitness
                self.best_violations = self.gbest_violation

    def report(self):
        print("================================")
        print("Solución Óptima")
        print("Individuo:", self.gbest_individual)
        print("Aptitud (Fitness):", self.gbest_fitness)
        print("Num Violaciones:", self.gbest_violation)
        print("================================")

    def evolution(self, verbose: bool = True):
        for gen in tqdm(range(GENERATIONS), desc="Evolucionando"):
                        
            for i in range(SIZE_POPULATION):
                objective = self.population[i]
                mutant = self._mutation_operator_(i, gen)
                trial = self._crossover_operator_(objective, mutant)
                if self.centroid_method:
                    trial = self.bounds_constraints(trial, self.population, self.lower, self.upper, self.violations)
                elif self.beta_method:
                    trial = self.bounds_constraints(
                        trial, self.lower, self.upper, self.population
                    )
                elif self.evolutionary_method:
                    trial = self.bounds_constraints(
                        trial, self.lower, self.upper, self.gbest_individual
                    )
                elif self.ADS:
                    trial = self.bounds_constraints(trial, self.lower, self.upper, self.gbest_individual, gen, GENERATIONS)
                elif self.resrand_method:
                    ...
                else:
                    trial = self.bounds_constraints(self.upper, self.lower, trial)

                self._selection_operator_(i, trial)

            if self.interactive:
                update_plot(
                    self.population,
                    self.fitness,
                    self.violations,
                    np.array([self.gbest_individual]),
                    gen,
                    self.scatter_plot_factible,
                    self.scatter_plot_infactible,
                    self.scatter_plot_gbest,
                    self.ax,
                )
            self.update_position_gbest_population()

        if verbose:
            self.report()

        plt.ioff()
        plt.show()
