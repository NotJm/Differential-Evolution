from typing import Callable, Tuple, List, Dict, Any
from algorithm import Algorithm
from constraints_functions import ConstriantsFunctionsHandler
from utils.constants import SIZE_POPULATION, GENERATIONS
from utils.check_pause import check_for_pause
from tqdm import tqdm
from mutation_strategy import MutationStrategies
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
        strategy: str = "rand1",
        centroid=False
    ):
        

        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions
        self.solutions_generate = []

        self.strategy = strategy
        self.centroid = centroid
        
        self.population = self.generate(self.upper, self.lower)
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        self.bounds_constraints = bounds_constraints
        self._compute_fitness_and_violations_()

        self._get_gbest_pobulation_zero_()

        self.mutation_strategies = MutationStrategies(self.population, self.F)

    def _compute_fitness_and_violations_(self):
        for index, individual in enumerate(self.population):
            fitness = self.objective_function(individual)
            self.fitness[index] = fitness

            total_de_violaciones = ConstriantsFunctionsHandler.sum_of_violations(
                self.g_functions, self.h_functions, individual
            )
            self.violations[index] = total_de_violaciones

    def _mutation_operator_(self, idx):

        samples = np.random.choice(SIZE_POPULATION, 5, replace=False)
        if self.strategy == "best1":
            return self.mutation_strategies._best1(samples)
        elif self.strategy == "rand1":
            return self.mutation_strategies._rand1(samples)
        elif self.strategy == "randtobest1":
            return self.mutation_strategies._randtobest1(samples)
        elif self.strategy == "currenttobest1":
            return self.mutation_strategies._currenttobest1(idx, samples)
        elif self.strategy == "best2":
            return self.mutation_strategies._best2(samples)
        elif self.strategy == "rand2":
            return self.mutation_strategies._rand2(samples)
        elif self.strategy == "res&ran":
            return self.mutation_strategies.res_rand(idx, self.lower, self.upper)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

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

    def report(self):
        print("================================")
        print("Solución Óptima")
        print("Individuo:", self.gbest_individual)
        print("Aptitud (Fitness):", self.gbest_fitness)
        print("Num Violaciones:", self.gbest_violation)
        print("================================")

    def evolution(self, verbose: bool = True):
        for _ in tqdm(range(GENERATIONS), desc="Evolucionando"):

            check_for_pause()

            for i in range(SIZE_POPULATION):
                objective = self.population[i]
                mutant = self._mutation_operator_(i)
                trial = self._crossover_operator_(objective, mutant)
                if self.centroid:
                    trial = self.bounds_constraints(trial, self.population, self.lower, self.upper, K=3)
                else:
                    trial = self.bounds_constraints(self.upper, self.lower, trial)
                self._selection_operator_(i, trial)

            self.update_position_gbest_population()

        if verbose:
            self.report()
