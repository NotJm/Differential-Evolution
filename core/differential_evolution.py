import numpy as np
import csv
from typing import Callable, Tuple, List
from .algorithm import Algorithm
from .constraints_functions import ConstriantsFunctionsHandler
from .bchms import BCHM
from utils.constants import SIZE_POPULATION, GENERATIONS
from tqdm import tqdm


class Differential_Evolution(Algorithm):

    def __init__(
        self,
        problem,
        objective_function: Callable,
        bounds_constraints_method: str,
        constraints_functions: Callable,
        bounds: Tuple[List, List] = ([], []),
        g_functions: List[Callable] = [],
        h_functions: List[Callable] = [],
        F: float = 0.7,
        CR: float = 0.9,
    ):
        self.problem = problem

        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions
        self.SFS = []
        self.SIS = []
        self.gbest_fitness_list = []
        self.gbest_violations_list = []

        self.population = self.generate(self.upper, self.lower)
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        
        self._compute_fitness_and_violations_()
        self._get_gbest_pobulation_zero_()

        
        self.method = bounds_constraints_method

    def _select_method_bounds_constriants_(self):
        BCHMS = [
            # "evo&cen"
            # "random_all",
            # "random_component",
            # "boundary",
            # "reflection",
            # "wrapping",
            # "centroid",
            # "evolutionary",
            # "beta",
            # "vector_wise_correction",
            "res&rand"
        ]

        method = np.random.choice(BCHMS)

        return method

    def _bounds_constraints_(self, method, idx, trial):
        if method == "random_all":
            trial = BCHM.random_all(self.upper, self.lower, trial)
        elif method == "random_component":
            trial = BCHM.random_component(self.upper, self.lower, trial)
        elif method == "boundary":
            trial = BCHM.boundary(self.upper, self.lower, trial)
        elif method == "reflection":
            trial = BCHM.reflection(self.upper, self.lower, trial)
        elif method == "wrapping":
            trial = BCHM.wrapping(self.upper, self.lower, trial)
        elif method == "centroid":
            trial = BCHM.centroid(
                trial,
                self.population,
                self.lower,
                self.upper,
                self.SFS,
                self.SIS,
                self.gbest_individual,
            )
        elif method == "evolutionary":
            trial = BCHM.evolutionary(
                trial, self.lower, self.upper, self.gbest_individual
            )
        elif method == "beta":
            trial = BCHM.beta(trial, self.lower, self.upper, self.population)
        elif method == "vector_wise_correction":
            trial = BCHM.vector_wise_correction(trial, self.upper, self.lower)
        elif method == "res&rand":
            trial = self.res_and_rand(idx)
        elif method == "AFC":
            trial = BCHM.evo_cen(trial, self.population, self.lower, self.upper, self.SIS, self.SFS, self.gbest_individual)

        return trial

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

    def _compute_diversity_(self):
        centroid = np.mean(self.population, axis=0)

        distances = np.linalg.norm(self.population - centroid, axis=1)

        std_deviation_distances = np.std(distances)

        return std_deviation_distances

    def _compute_percentage_factibility_(self):
        num_factible = np.count_nonzero(self.violations == 0)
        try:
            percentage_factibility = num_factible / SIZE_POPULATION
            return percentage_factibility
        except ZeroDivisionError:
            return 0

    def _compute_percentage_generation_(self, gen):
        percentage_generation = gen / GENERATIONS
        return percentage_generation

    def _compare_objective_with_trial_(self, objective, trial):
        trial_fitness = self.objective_function(trial)
        trial_violations = ConstriantsFunctionsHandler.sum_of_violations(
            self.g_functions, self.h_functions, trial
        )

        objective_fitness = self.objective_function(objective)
        objective_violations = ConstriantsFunctionsHandler.sum_of_violations(
            self.g_functions, self.h_functions, objective
        )

        if not self.constraints_functions(
            objective_fitness, objective_violations, trial_fitness, trial_violations
        ):
            return 1
        else:
            return 0

    def _mutation_operator_(self, idx):
        index = np.arange(len(self.population))
        index = np.delete(index, idx)

        r1, r2, r3 = np.random.choice(index, 3, replace=False)

        X_r1 = self.population[r1]
        X_r2 = self.population[r2]
        X_r3 = self.population[r3]

        mutado = X_r1 + self.F * (X_r2 - X_r3)

        return mutado
    
    def res_and_rand(self, idx, max_resamples=3):
        NP, D = self.population.shape
        no_res = 0
        valid = False
        while no_res < max_resamples * D and not valid:
            indices = np.arange(NP)
            indices = np.delete(indices, idx)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)

            V = self.population[r1] + self.F * (
                self.population[r2] - self.population[r3]
            )

            valid = self.isValid(self.upper, self.lower, V)
            no_res += 1

        if not valid:
            V = BCHM.random_component(self.upper, self.lower, V)

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
        gbest_position_initial = 0

        self.gbest_fitness = self.fitness[gbest_position_initial]
        self.gbest_violation = self.violations[gbest_position_initial]
        self.gbest_individual = self.population[gbest_position_initial]

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
                
                # Store gbest values for plotting
                self.gbest_fitness_list.append(self.gbest_fitness)
                self.gbest_violations_list.append(self.gbest_violation)

    def report(self, method, winner, diversity, percentage_factibility):
        print("================================")
        print("Selected Method:", method)
        print("Optimal Solution")
        print("Individual:", self.gbest_individual)
        print("Diversity:", diversity)
        print("Percentage Factibility for Population:",percentage_factibility)
        print("Trial is best" if winner == 1 else "Target is best")
        print("Fitness:", self.gbest_fitness)
        print("Violation:", self.gbest_violation)
        print("================================")

    def _rewrite_csv_(
        self, method, diversity, percentage_factibility, percentage_generations, best
    ):
        # Registrar datos en el archivo CSV
        with open(self.csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    method,
                    diversity,  # Diversidad actual
                    percentage_factibility,  # Porcentaje de factibilidad
                    percentage_generations,
                    best,
                    self.problem,
                ]
            )

    def evolution(self, verbose: bool = True):
        for gen in tqdm(range(1, GENERATIONS + 1), desc="Evolucionando"):
            # Calculo de SFS y SIS
            self._compute_SFS_SIS_()

            # Calculo de diversidad
            # diversity = self._compute_diversity_()
            # Calculo del porcentaje de factibilidad
            # percentage_factibility = self._compute_percentage_factibility_()
            # Calculo del porcentaje de generaciones
            # percentage_generations = self._compute_percentage_generation_(gen)

            for i in range(SIZE_POPULATION):
               
                
                objective = self.population[i]
                mutant = self._mutation_operator_(i)
                trial = self._crossover_operator_(objective, mutant)

                if not self.isValid(self.upper, self.lower, trial):
                    # Seleccion de metodos
                    # method = self._select_method_bounds_constriants_()
                     
                    trial = self._bounds_constraints_(self.method, i, trial)

                    # Comparacion de objectivo con el de prueba
                    # best = self._compare_objective_with_trial_(objective, trial)
                          
                self._selection_operator_(i, trial)
            
            self.update_position_gbest_population()
                 
           

            
