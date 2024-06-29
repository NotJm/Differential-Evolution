import inquirer
import pandas as pd
from core.boundary_handler import BoundaryHandler
from functions.cec2020problems import *
from functions.cec2006problems import *
from functions.cec2017problems import *
from functions.cec2010problems import *
from functions.cec2006problems import *
from utils.helpers import (
    execute_algorithm,
    plot_convergence_violations_all,
    plot_results_all,
    save_results_to_csv,
    plot_fitness_boxplot_all,
    generate_random_filename,
    plot_fitness_boxplot_from_csvs,
    plot_violations_boxplot_from_csvs,
)

# ! CEC2006
"""problems = {
    "G01": CEC2006_G01,
    "G02": CEC2006_G02,
    "G03": CEC2006_G03,
    "G04": CEC2006_G04,
    "G05": CEC2006_G05,
    "G06": CEC2006_G06,
    "G07": CEC2006_G07,
    "G08": CEC2006_G08,
    "G09": CEC2006_G09,
    "G10": CEC2006_G10,
    "G11": CEC2006_G11,
    "G12": CEC2006_G12,
    "G13": CEC2006_G13,
    "G14": CEC2006_G14,
    "G15": CEC2006_G15,
    "G17": CEC2006_G17,
    "G18": CEC2006_G18,
    "G19": CEC2006_G19,
    "G20": CEC2006_G20,
    "G21": CEC2006_G21,
    "G22": CEC2006_G22,
    "G23": CEC2006_G23,
    "G24": CEC2006_G24,
}"""
# ! CEC2017
problems = {
    # "C01": CEC2017_C01,
    # "C02": CEC2017_C02,
    # "C03": CEC2017_C03,
    # "C04": CEC2017_C04,
    # "C05": CEC2017_C05,
    # "C06": CEC2017_C06,
    # "C07": CEC2017_C07,
    # "C08": CEC2017_C08,
    # "C09": CEC2017_C09,
    # "C10": CEC2017_C10,
    # "C11": CEC2017_C11,
    # "C12": CEC2017_C12,
    # "C13": CEC2017_C13,
    # "C14": CEC2017_C14,
    # "C15": CEC2017_C15,
    # "C16": CEC2017_C16,
    # "C17": CEC2017_C17,
    # "C18": CEC2017_C18,
    # "C19": CEC2017_C19,
    # "C20": CEC2017_C20,
    # "C21": CEC2017_C21,
    # "C22": CEC2017_C22,
    # "C23": CEC2017_C23,
    # "C24": CEC2017_C24,
    # "C25": CEC2017_C25,
    # "C26": CEC2017_C26,
    # "C27": CEC2017_C27,
    # "C28": CEC2017_C28,
}

problems = {
    "C01": CEC2010_C01,
    "C02": CEC2010_C02,
    "C03": CEC2010_C03,
    "C04": CEC2010_C04,
    "C05": CEC2010_C05,
    "C06": CEC2010_C06,
    "C07": CEC2010_C07,
    "C08": CEC2010_C08,
    "C09": CEC2010_C09,
    "C10": CEC2010_C10,
    "C11": CEC2010_C11,
    "C12": CEC2010_C12,
    "C13": CEC2010_C13,
    "C14": CEC2010_C14,
    "C15": CEC2010_C15,
    "C16": CEC2010_C16,
    "C17": CEC2010_C17,
    "C18": CEC2010_C18,
}

bounds = {
        "reflection": BoundaryHandler.agarwl_reflect,
        "beta": BoundaryHandler.andreaa_beta,
        "boundary": BoundaryHandler.andreaa_saturation,
        "random": BoundaryHandler.andreaa_uniform,
        "vector_wise_correction": BoundaryHandler.andreaa_vector_wise_correction,
        "evolutionary": BoundaryHandler.gandomi_evolutionary,
        "wrapping": BoundaryHandler.qin_wrapping,
        "centroid": BoundaryHandler.centroid_method,
        "dynamic_correction": BoundaryHandler.dynamic_correction,
        "res&rand": None,
}

def run():

    def select_options():
        choices = [
            "Ejecutar todas las restricciones por problema",
            "Elegir un problema y restricción específicos",
        ]
        questions = [
            inquirer.List("choice", message="¿Qué te gustaría hacer?", choices=choices)
        ]
        answers = inquirer.prompt(questions)
        return answers["choice"]

    def select_problem_and_constraint():
        problem_questions = [
            inquirer.List(
                "problem",
                message="Selecciona un problema",
                choices=list(problems.keys()),
            )
        ]
        selected_problem = inquirer.prompt(problem_questions)["problem"]

        constraint_questions = [
            inquirer.List(
                "constraint",
                message="Selecciona una restricción",
                choices=list(bounds.keys()),
            )
        ]
        selected_constraint = inquirer.prompt(constraint_questions)["constraint"]

        return selected_problem, selected_constraint

    choice = select_options()

    results = []

    if choice == "Ejecutar todas las restricciones por problema":
        for problem_name, problem_class in problems.items():
            all_convergence_data = {}
            all_violations_data = {}
            all_fitness_data = {}
            for constraint_name in bounds:
                (
                    fitness_data,
                    violations_data,
                    convergence_violations_data,
                    factible_fitness_data,
                ) = execute_algorithm(
                    problem_name, problem_class, constraint_name, bounds
                )

                all_convergence_data[constraint_name] = convergence_violations_data
                all_violations_data[constraint_name] = violations_data
                all_fitness_data[constraint_name] = factible_fitness_data

                average_fitness = (
                    sum(factible_fitness_data) / len(factible_fitness_data)
                    if factible_fitness_data
                    else "N/A"
                )
                results.append(
                    [
                        problem_name,
                        constraint_name,
                        average_fitness,
                        len(factible_fitness_data),
                    ]
                )

            plot_convergence_violations_all(problem_name, all_convergence_data, bounds)
            plot_results_all(problem_name, all_violations_data, bounds)
            plot_fitness_boxplot_all(problem_name, all_fitness_data, bounds)
    else:
        problem_name, constraint_name = select_problem_and_constraint()
        (
            fitness_data,
            violations_data,
            convergence_violations_data,
            factible_fitness_data,
        ) = execute_algorithm(
            problem_name, problems[problem_name], constraint_name, bounds
        )

        all_convergence_data = {constraint_name: convergence_violations_data}
        plot_convergence_violations_all(problem_name, all_convergence_data, bounds)

        average_fitness = (
            sum(factible_fitness_data) / len(factible_fitness_data)
            if factible_fitness_data
            else "N/A"
        )
        results.append(
            [problem_name, constraint_name, average_fitness, len(factible_fitness_data)]
        )

        all_violations_data = {constraint_name: violations_data}
        plot_results_all(problem_name, all_violations_data, bounds)

        all_fitness_data = {constraint_name: factible_fitness_data}
        plot_fitness_boxplot_all(problem_name, all_fitness_data, bounds)

    # Convert results to DataFrame format expected in save_results_to_excel
    results_df = pd.DataFrame(
        results,
        columns=[
            "Problema",
            "Restricción",
            "Fitness Promedio",
            "Ejecuciones Factibles",
        ],
    )

    save_results_to_csv(results_df, generate_random_filename())


if __name__ == "__main__":
    run()
