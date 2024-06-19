import gc
import os
import matplotlib.pyplot as plt
import pandas as pd
import inquirer
from core.boundary_handler import BoundaryHandler
from core.constraints_functions import ConstriantsFunctionsHandler
from utils.constants import EXECUTIONS
from core.differential_evolution import Differential_Evolution
from functions.cec2020problems import *
from functions.cec2006problems import *

def run():
    
    problems = {
        "R01": CEC2020_RC01,
        "R02": CEC2020_RC02,
        "R03": CEC2020_RC03,
        "R05": CEC2020_RC05,
        "R06": CEC2020_RC06,
    }

    bounds = {
        # Agarwl 2016
        "reflection": BoundaryHandler.agarwl_reflect,
        # Andrea 2023
        "beta": BoundaryHandler.andreaa_beta,
        # Andrea 2023
        "boundary": BoundaryHandler.andreaa_saturation,
        # Andrea 2023
        "random": BoundaryHandler.andreaa_uniform,
        # Andrea 2023
        "vector_wise_correction": BoundaryHandler.andreaa_vector_wise_correction,
        # Gandomi 2012
        "evolutionary": BoundaryHandler.gandomi_evolutionary,
        # Qin 2009
        "wrapping": BoundaryHandler.qin_wrapping,    
        # Juarez 2019
        "centroid": BoundaryHandler.centroid_method,
        # Juarez 2019
        "res_and_rand": None
    }

    def execute_algorithm(problem_name, problem_class, constraint_name):
        problema = problem_class()
        fitness_data = []
        violations_data = []
        convergence_violations_data = []

        # Ejecucion del algoritmo por problema y restriccion
        for _ in range(EXECUTIONS):
            print(f"Ejecucion {_ + 1} para problema {problem_name} con el metodo {constraint_name}:")
            try:
                # Algoritmo de evolucion diferencial
                algorithm = Differential_Evolution(
                    problema.fitness,
                    ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                    bounds_constraints=bounds[constraint_name],
                    bounds=(problema.superior, problema.inferior),
                    g_functions=problema.rest_g,
                    h_functions=problema.rest_h,
                    centroid=(constraint_name == "centroid"),
                    beta=(constraint_name == "beta"),
                    evolutionary=(constraint_name == "evolutionary"),
                    res_and_rand=(constraint_name == "res_and_rand"),
                )

                # Mostrar informacion sobre las soluciones optimas
                algorithm.evolution(verbose=True)
                
                # Guardando datos de fitness y violaciones
                fitness_data.append(algorithm.gbest_fitness)
                violations_data.append(algorithm.gbest_violation)
                convergence_violations_data.append(algorithm.gbest_violations_list)

            except Exception as e:
                # En caso de error solo prosigue
                print(f"Error en el metodo {constraint_name} en ejecución {_ + 1}: {e}")

            # Activando el garbage collector
            del algorithm
            gc.collect()

        # Filtrar ejecuciones factibles
        factible_indices = [i for i, v in enumerate(violations_data) if v == 0]
        factible_fitness_data = [fitness_data[i] for i in factible_indices]
        
        return fitness_data, violations_data, convergence_violations_data, factible_fitness_data

    def plot_convergence_violations_all(problem_name, all_convergence_data):
        # Grafico de convergencia de violaciones para todas las restricciones en un problema
        plt.figure(figsize=(10, 6))
        for constraint_name, convergence_data in all_convergence_data.items():
            for run_data in convergence_data:
                plt.plot(run_data, alpha=0.5, label=constraint_name if run_data == convergence_data[0] else "")
        plt.title(f'Convergence Plot (Violations) for {problem_name}')
        plt.xlabel('Generations')
        plt.ylabel('Violations')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_results_all(problem_name, all_violations_data):
        # BoxPlot de Violaciones para todas las restricciones en un problema
        plt.figure(figsize=(12, 8))
        data = [violations for violations in all_violations_data.values()]
        labels = [constraint for constraint in all_violations_data.keys()]
        plt.boxplot(data, labels=labels)
        plt.title(f'Box Plot de Violations para {problem_name}')
        plt.xlabel('Restricción')
        plt.ylabel('Violations')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def select_options():
        choices = ["Ejecutar todas las restricciones por problema", "Elegir un problema y restricción específicos"]
        questions = [inquirer.List('choice', message="¿Qué te gustaría hacer?", choices=choices)]
        answers = inquirer.prompt(questions)
        return answers['choice']

    def select_problem_and_constraint():
        problem_questions = [inquirer.List('problem', message="Selecciona un problema", choices=list(problems.keys()))]
        selected_problem = inquirer.prompt(problem_questions)['problem']

        constraint_questions = [inquirer.List('constraint', message="Selecciona una restricción", choices=list(bounds.keys()))]
        selected_constraint = inquirer.prompt(constraint_questions)['constraint']

        return selected_problem, selected_constraint

    choice = select_options()

    results = []

    if choice == "Ejecutar todas las restricciones por problema":
        for problem_name, problem_class in problems.items():
            all_convergence_data = {}
            all_violations_data = {}
            for constraint_name in bounds:
                fitness_data, violations_data, convergence_violations_data, factible_fitness_data = execute_algorithm(problem_name, problem_class, constraint_name)
                
                # Almacenar datos de convergencia de violaciones
                all_convergence_data[constraint_name] = convergence_violations_data

                # Almacenar datos de violaciones
                all_violations_data[constraint_name] = violations_data

                # Calcular promedio de fitness para ejecuciones factibles
                if factible_fitness_data:
                    average_fitness = sum(factible_fitness_data) / len(factible_fitness_data)
                else:
                    average_fitness = "N/A"

                results.append([problem_name, constraint_name, average_fitness, len(factible_fitness_data)])

            # Plot de convergencia de violaciones para todas las restricciones en el problema actual
            plot_convergence_violations_all(problem_name, all_convergence_data)
            
            # Plot de BoxPlot de violaciones para todas las restricciones en el problema actual
            plot_results_all(problem_name, all_violations_data)
    else:
        problem_name, constraint_name = select_problem_and_constraint()
        fitness_data, violations_data, convergence_violations_data, factible_fitness_data = execute_algorithm(problem_name, problems[problem_name], constraint_name)
        
        # Almacenar datos de convergencia de violaciones
        all_convergence_data = {constraint_name: convergence_violations_data}

        # Plot de convergencia de violaciones
        plot_convergence_violations_all(problem_name, all_convergence_data)

        # Calcular promedio de fitness para ejecuciones factibles
        if factible_fitness_data:
            average_fitness = sum(factible_fitness_data) / len(factible_fitness_data)
        else:
            average_fitness = "N/A"

        results.append([problem_name, constraint_name, average_fitness, len(factible_fitness_data)])

        # Almacenar datos de violaciones
        all_violations_data = {constraint_name: violations_data}
        
        # Plot de BoxPlot de violaciones
        plot_results_all(problem_name, all_violations_data)

    # Crear DataFrame y exportar a Excel
    df_results = pd.DataFrame(results, columns=["Problema", "Restricción", "Fitness Promedio", "Ejecuciones Factibles"])
    df_results.to_excel("resultados.xlsx", index=False)

if __name__ == '__main__':
    run()
