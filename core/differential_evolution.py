from typing import Callable, Tuple, List
from .algorithm import Algorithm
from .constraints_functions import ConstriantsFunctionsHandler
from utils.constants import SIZE_POPULATION, GENERATIONS
from utils.check_pause import check_for_pause
from tqdm import tqdm
from .mutation_strategy import MutationStrategies
import numpy as np
import time


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
        centroid: bool = False,
        beta: bool = False,
        evolutionary: bool = False,
        res_and_rand: bool = False,
        dynamic_correction: bool = False,
    ):

        self.F = F
        self.CR = CR
        self.upper, self.lower = bounds
        self.g_functions = g_functions
        self.h_functions = h_functions
        self.solutions_generate = []

        self.strategy = strategy
        self.centroid = centroid
        self.beta = beta
        self.evolutionary = evolutionary
        self.res_and_rand = res_and_rand
        self.dynamic_correction = dynamic_correction

        # Lists to store gbest values for plotting convergence
        self.gbest_fitness_list = []
        self.gbest_violations_list = []

        self.population = self.generate(self.upper, self.lower)
        self.NP, self.D = self.population.shape
        self.fitness = np.zeros(SIZE_POPULATION)
        self.violations = np.zeros(SIZE_POPULATION)
        self.objective_function = objective_function
        self.constraints_functions = constraints_functions
        self.bounds_constraints = bounds_constraints
        self._compute_fitness_and_violations_()

        self._get_gbest_pobulation_zero_()

        self.mutation_strategies = MutationStrategies(self.population, self.F, self.objective_function)

    def _compute_fitness_and_violations_(self):
        for index, individual in enumerate(self.population):
            fitness = self.objective_function(individual)
            self.fitness[index] = fitness

            total_de_violaciones = ConstriantsFunctionsHandler.sum_of_violations(
                self.g_functions, self.h_functions, individual
            )
            self.violations[index] = total_de_violaciones
    
    def _mutation_operator_(self, idx, generation=0):
        
        if self.res_and_rand:
            self.res_and_ran_mutation(idx)
        
        if self.strategy in ['rand3', 'best3']:
            samples = np.random.choice(SIZE_POPULATION, 6, replace=False)
        elif self.strategy == 'adaptive_rand_elite':
            return self.mutation_strategies._adaptive_rand_elite(generation)
        else:
            samples = np.random.choice(SIZE_POPULATION, 5, replace=False)
            
        if self.strategy == 'best1':
            return self.mutation_strategies._best1(samples)
        elif self.strategy == 'rand1':
            return self.mutation_strategies._rand1(samples)
        elif self.strategy == 'randtobest1':
            return self.mutation_strategies._randtobest1(samples)
        elif self.strategy == 'currenttobest1':
            return self.mutation_strategies._currenttobest1(idx, samples)
        elif self.strategy == 'best2':
            return self.mutation_strategies._best2(samples)
        elif self.strategy == 'rand2':
            return self.mutation_strategies._rand2(samples)
        elif self.strategy == 'currenttorand1': 
            return self.mutation_strategies._currenttorand1(idx, samples)
        elif self.strategy == 'best3':
            return self.mutation_strategies._best3(samples)
        elif self.strategy == 'rand3':
            return self.mutation_strategies._rand3(samples)
        elif self.strategy == 'randtocurrent2':
            return self.mutation_strategies._rand_to_current2(idx, samples)
        elif self.strategy == 'randToBestAndCurrent2':
            return self.mutation_strategies._rand_to_best_and_current2(idx, samples)
        elif self.strategy == 'combined_100%':
            return self.mutation_strategies.combined_rand1_best1(0.100)
        elif self.strategy == 'combined_95%':
            return self.mutation_strategies.combined_rand1_best1(0.95)
        elif self.strategy == 'combined_90%':
            return self.mutation_strategies.combined_rand1_best1(0.90)
        elif self.strategy == 'combined_85%':
            return self.mutation_strategies.combined_rand1_best1(0.85)
        elif self.strategy == 'combined_80%':
            return self.mutation_strategies.combined_rand1_best1(0.80)
        elif self.strategy == 'combined_75%':
            return self.mutation_strategies.combined_rand1_best1(0.75)
        elif self.strategy == 'combined_70%':
            return self.mutation_strategies.combined_rand1_best1(0.70)
        elif self.strategy == 'combined_65%':
            return self.mutation_strategies.combined_rand1_best1(0.65)
        elif self.strategy == 'combined_60%':
            return self.mutation_strategies.combined_rand1_best1(0.60)
        elif self.strategy == 'combined_55%':
            return self.mutation_strategies.combined_rand1_best1(0.55)
        elif self.strategy == 'combined_50%':
            return self.mutation_strategies.combined_rand1_best1(0.50)
        elif self.strategy == 'combined_45%':
            return self.mutation_strategies.combined_rand1_best1(0.45)
        elif self.strategy == 'combined_40%':
            return self.mutation_strategies.combined_rand1_best1(0.40)
        elif self.strategy == 'combined_35%':
            return self.mutation_strategies.combined_rand1_best1(0.35)
        elif self.strategy == 'combined_30%':
            return self.mutation_strategies.combined_rand1_best1(0.30)
        elif self.strategy == 'combined_25%':
            return self.mutation_strategies.combined_rand1_best1(0.25)
        elif self.strategy == 'combined_20%':
            return self.mutation_strategies.combined_rand1_best1(0.20)
        elif self.strategy == 'combined_15%':
            return self.mutation_strategies.combined_rand1_best1(0.15)
        elif self.strategy == 'combined_10%':
            return self.mutation_strategies.combined_rand1_best1(0.10)
        elif self.strategy == 'combined_05%':
            return self.mutation_strategies.combined_rand1_best1(0.05)
        elif self.strategy == 'combined_0%':
            return self.mutation_strategies.combined_rand1_best1(0.0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
             
    def res_and_ran_mutation(self, i):
        F = 0.8  # Factor de escala para la mutación
        attempts = 3 * self.D  # Máximo de intentos de remuestreo
        
        indices_pool = np.delete(np.arange(self.NP), i)
        valid_V = None

        for _ in range(attempts):
            indices = np.random.choice(indices_pool, 3, replace=False)
            r1, r2, r3 = self.population[indices]
            V = r1 + F * (r2 - r3)

            # Verificar si el vector mutante está dentro de los límites
            if np.all((self.lower <= V) & (V <= self.upper)):
                valid_V = V
                break

        # Si no se obtuvo un vector válido después del máximo de remuestreos
        if valid_V is None:
            valid_V = np.random.uniform(self.lower, self.upper, self.D)

        return valid_V
        

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

                
                
                # Store gbest values for plotting
                self.gbest_fitness_list.append(self.gbest_fitness)
                self.gbest_violations_list.append(self.gbest_violation)
                
                self.best_fitness = self.gbest_fitness
                self.best_violations = self.gbest_violation

                self.best_fitness = self.gbest_fitness
                self.best_violations = self.gbest_violation

    def report(self):
        start_time = time.time()

        # Calcular estadísticas
        mean_fitness = np.mean(self.gbest_fitness_list)
        std_fitness = np.std(self.gbest_fitness_list)
        mean_violations = np.mean(self.gbest_violations_list)
        std_violations = np.std(self.gbest_violations_list)

        end_time = time.time()
        execution_time = end_time - start_time

        print("================================")
        print("Solución Óptima")
        print("Individuo:", self.gbest_individual)
        print("Aptitud (Fitness):", self.gbest_fitness)
        print("Num Violaciones:", self.gbest_violation)
        print("================================")
        print("Estadísticas de Convergencia")
        print(f"Media de Fitness: {mean_fitness}")
        print(f"Desviación Estándar de Fitness: {std_fitness}")
        print(f"Media de Violaciones: {mean_violations}")
        print(f"Desviación Estándar de Violaciones: {std_violations}")
        print(f"Tiempo de Ejecución del Reporte: {execution_time} segundos")
        print("================================")

    def evolution(self, verbose: bool = True):
        for _ in tqdm(range(GENERATIONS), desc="Evolucionando"):

            # check_for_pause(self.report)

            for i in range(SIZE_POPULATION):
                objective = self.population[i]
                mutant = self._mutation_operator_(i)
                trial = self._crossover_operator_(objective, mutant)
                if self.centroid:
                    trial = self.bounds_constraints(
                        trial, self.population, self.lower, self.upper, K=3
                    )
                elif self.beta:
                    trial = self.bounds_constraints(
                        trial, self.lower, self.upper, self.population
                    )
                elif self.evolutionary:
                    trial = self.bounds_constraints(
                        trial, self.lower, self.upper, self.gbest_individual
                    )
                elif self.res_and_rand:
                    pass
                elif self.dynamic_correction:
                    trial = self.bounds_constraints(trial, self.population, self.lower, self.upper)
                else:
                    trial = self.bounds_constraints(self.upper, self.lower, trial)
                self._selection_operator_(i, trial)

            self.update_position_gbest_population()

        if verbose:
            self.report()

    
    # def evolution(self, verbose: bool = True):
    #     for gen in tqdm(range(GENERATIONS), desc="Evolucionando"):
    #        for i in range(SIZE_POPULATION):
    #             objective = self.population[i]
    #             mutant = self.mutation_operator(i, gen)  # Pasando 'generation' aquí
    #             trial = self._crossover_operator_(objective, mutant)
    #             if self.centroid:
    #                 trial = self.bounds_constraints(
    #                     trial, self.population, self.lower, self.upper
    #                 )
    #             else:
    #                 trial = self.bounds_constraints(self.upper, self.lower, trial)
    #             self._selection_operator_(i, trial)

    #     self.update_position_gbest_population()

    #     if verbose:
    #         self.report()

    # def evolution(self, verbose: bool = True):
    #     for gen in tqdm(range(GENERATIONS), desc="Evolucionando"):
    #        for i in range(SIZE_POPULATION):
    #             objective = self.population[i]
    #             mutant = self._mutation_operator_(i, gen)  # Pasando 'generation' aquí
    #             trial = self._crossover_operator_(objective, mutant)
    #             if self.centroid:
    #                 trial = self.bounds_constraints(
    #                     trial, self.population, self.lower, self.upper
    #                 )
    #             else:
    #                 trial = self.bounds_constraints(self.upper, self.lower, trial)
    #             self._selection_operator_(i, trial)

    #     self.update_position_gbest_population()

       # graficar_convergencia(
    #     #     self.solutions_generate,
    #     #     "reportMutation/problemar02.png",
    #     #     " -DE/rand1/1 - ",
    #     # )

        # if verbose:
        #     self.report()

        

       
