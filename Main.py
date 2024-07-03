import pandas as pd
from core.boundary_handler import BoundaryHandler
from functions.cec2020problems import *
from functions.cec2006problems import *
from functions.cec2017problems import *
from functions.cec2010problems import *
from functions.cec2006problems import *
from utils.execute import execute_algorithm
from utils.plotting import (
    plot_fitness_boxplot_all,
    plot_convergence_violations_all,
    plot_violations_boxplot_all,
)
from utils.save_file import save_results_to_csv


def main(problems, bchms, directory, problem_prefix):

    results = []

    for problem_name, problem_class in problems.items():
        all_convergence_data = {}
        all_violations_data = {}
        all_fitness_data = {}
        for bchm in bchms:
            (
                fitness_data,
                violations_data,
                convergence_violations_data,
                factible_fitness_data,
            ) = execute_algorithm(
                problem_name, problem_class, bchm, bchms, DIRECTORY, PROBLEM_PREFIX
            )

            all_convergence_data[bchm] = convergence_violations_data
            all_violations_data[bchm] = violations_data
            all_fitness_data[bchm] = factible_fitness_data

            average_fitness = (
                sum(factible_fitness_data) / len(factible_fitness_data)
                if factible_fitness_data
                else "N/A"
            )
            results.append(
                [
                    problem_name,
                    bchm,
                    average_fitness,
                    len(factible_fitness_data),
                ]
            )

            
            plot_violations_boxplot_all(
                problem_name, all_violations_data, directory, problem_prefix
            )
            plot_fitness_boxplot_all(
                problem_name, all_fitness_data , directory, problem_prefix
            )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Problema",
            "Restricción",
            "Fitness Promedio",
            "Ejecuciones Factibles",
        ],
    )

    save_results_to_csv(results_df, directory, problem_prefix)


if __name__ == "__main__":
    problems = {
        "C01": CEC2010_C01,
        # "C02": CEC2010_C02,
        # "C03": CEC2010_C03,
        # "C04": CEC2010_C04,
        # "C05": CEC2010_C05,
        # "C06": CEC2010_C06,
        # "C07": CEC2010_C07,
        # "C08": CEC2010_C08,
        # "C09": CEC2010_C09,
        # "C10": CEC2010_C10,
        # "C11": CEC2010_C11,
        # "C12": CEC2010_C12,
        # "C13": CEC2010_C13,
        # "C14": CEC2010_C14,
        # "C15": CEC2010_C15,
        # "C16": CEC2010_C16,
        # "C17": CEC2010_C17,
        # "C18": CEC2010_C18,
    }

    bchms = {
        "resampling": BoundaryHandler.scalar_compression_method,
        # "reflection": BoundaryHandler.agarwl_reflect,
        # "beta": BoundaryHandler.andreaa_beta,
        # "boundary": BoundaryHandler.andreaa_saturation,
        # "random": BoundaryHandler.andreaa_uniform,
        # "vector_wise_correction": BoundaryHandler.andreaa_vector_wise_correction,
        # "evolutionary": BoundaryHandler.gandomi_evolutionary,
        # "wrapping": BoundaryHandler.qin_wrapping,
        # "centroid": BoundaryHandler.centroid_method,
        # "res&rand": None,
    }

    DIRECTORY = "mnt/data/cec2010"
    PROBLEM_PREFIX = "CEC2010"

    main(problems, bchms, DIRECTORY, PROBLEM_PREFIX)
    
