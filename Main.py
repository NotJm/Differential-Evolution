import inquirer
import pandas as pd
from core.boundary_handler import BoundaryHandler
from functions.cec2020problems import *
from functions.cec2006problems import *
from functions.cec2017problems import *
from utils.helpers import (
    execute_algorithm,
    plot_convergence_violations_all,
    plot_results_all,
    save_results_to_csv,
    plot_fitness_boxplot_all,
    generate_random_filename,
    plot_fitness_boxplot_from_csvs,
    plot_violations_boxplot_from_csvs
)


def run():
    problems = {
        "C04": CEC2017_C04,
        "C05": CEC2017_C05,
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
        "res_and_rand": None,
    }

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
    plot_violations_boxplot_from_csvs("report/cec2017", "C03")
