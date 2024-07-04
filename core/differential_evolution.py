from typing import Callable, Tuple, List
from .algorithm import Algorithm
from .constraints_functions import ConstriantsFunctionsHandler
from .bchms import BCHM
from utils.constants import SIZE_POPULATION, GENERATIONS
from tqdm import tqdm
import numpy as np


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
        centroid_repair_method: bool = False,
        beta_method: bool = False,
        evolutionary_method: bool = False,
        resrand_method: bool = False,
        ADS: bool = False,
        ABB: bool = False,
    ):

        self.centroid_method = centroid_method
        self.centroid_repair_method = centroid_repair_method
        self.beta_method = beta_method
        self.evolutionary_method = evolutionary_method
        self.resrand_method = resrand_method
        self.ADS = ADS
        self.ABB = ABB

        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions
        self.SFS = []
        self.SIS = []

        self.population = self.generate(self.upper, self.lower)
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        self.bounds_constraints = bounds_constraints
        self._compute_fitness_and_violations_()

        self._get_gbest_pobulation_zero_()

    def _compute_fitness_and_violations_(self):
        for index, individual in enumerate(self.population):
            fitness = self.objective_function(individual)
            self.fitness[index] = fitness

            total_de_violaciones = ConstriantsFunctionsHandler.sum_of_violations(
                self.g_functions, self.h_functions, individual
            )
            self.violations[index] = total_de_violaciones

    def _compute_SFS_SIS_(self):
        self.SFS = np.where(self.violations == 0)[0]
        self.SIS = np.where(self.violations > 0)[0]

    def _mutation_operator_(self, idx):
        if not self.resrand_method:
            index = np.arange(len(self.population))
            index = np.delete(index, idx)

            r1, r2, r3 = np.random.choice(index, 3, replace=False)

            X_r1 = self.population[r1]
            X_r2 = self.population[r2]
            X_r3 = self.population[r3]

            mutado = X_r1 + self.F * (X_r2 - X_r3)

            return mutado
        else:
            return self.res_and_rand(idx)

    def res_and_rand(self, idx, max_resamples=3):
        NP, D = self.population.shape
        no_res = 0
        valid = False
        while no_res < max_resamples * D and not valid:
            indices = np.arange(NP)
            indices = np.delete(indices, idx)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)

            V = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

            valid = self.isValid(self.upper, self.lower, V)
            no_res += 1

        if not valid:
            BCHM.random_component(self.upper, self.lower, V)

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
        position_initial = 0

        self.gbest_fitness = self.fitness[position_initial]
        self.gbest_violation = self.violations[position_initial]
        self.gbest_individual = self.population[position_initial]

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

    def report(self):
        print("================================")
        print("Optimal Solution")
        print("Individual:", self.gbest_individual)
        print("Fitness:", self.gbest_fitness)
        print("Violation:", self.gbest_violation)
        print("================================")

    def evolution(self, verbose: bool = True):
        for gen in tqdm(range(GENERATIONS), desc="Evolucionando"):

            self._compute_SFS_SIS_()

            for i in range(SIZE_POPULATION):
                objective = self.population[i]
                mutant = self._mutation_operator_(i)
                trial = self._crossover_operator_(objective, mutant)

                if not self.isValid(self.upper, self.lower, trial):
                    if self.centroid_method:
                        trial = self.bounds_constraints(
                            trial,
                            self.population,
                            self.lower,
                            self.upper,
                            self.SFS,
                            self.SIS,
                            self.gbest_individual,
                        )
                    elif self.centroid_repair_method:
                        trial = self.bounds_constraints(
                            trial,
                            self.population,
                            self.lower,
                            self.upper,
                        )
                    elif self.beta_method:
                        trial = self.bounds_constraints(
                            trial, self.lower, self.upper, self.population
                        )
                    elif self.evolutionary_method:
                        trial = self.bounds_constraints(
                            trial, self.lower, self.upper, self.gbest_individual
                        )
                    elif self.ADS:
                        trial = self.bounds_constraints(
                            trial,
                            self.lower,
                            self.upper,
                            self.gbest_individual,
                            gen,
                            GENERATIONS,
                        )
                    elif self.resrand_method:
                        ...
                    else:
                        trial = self.bounds_constraints(self.upper, self.lower, trial)

                self._selection_operator_(i, trial)

            self.update_position_gbest_population()

        if verbose:
            self.report()
