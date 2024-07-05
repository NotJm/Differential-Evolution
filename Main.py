import pandas as pd
from core.boundary_handler import BoundaryHandler
from core.bchms import BCHM
from functions.cec2020problems import *
from functions.cec2006problems import *
from functions.cec2017problems_update import *
from functions.cec2010problems_update import *
from functions.cec2006problems import *
from utils.execute import execute_algorithm
from utils.plotting import (
    plot_fitness_boxplot_all,
    plot_convergence_violations_all,
    plot_violations_boxplot_all,
    plot_fitness_boxplot_from_csvs,
    plot_violations_boxplot_from_csvs,
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
                problem_name, all_fitness_data, directory, problem_prefix
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
        "C01": CEC2017_C01,
        "C02": CEC2017_C02,
        "C03": CEC2017_C03,
        "C04": CEC2017_C04,
        "C05": CEC2017_C05,
        "C06": CEC2010_C06,
        "C07": CEC2010_C07,
        "C08": CEC2010_C08,
        "C09": CEC2010_C09,
        "C10": CEC2017_C10,
        "C11": CEC2017_C11,
        "C12": CEC2017_C12,
        "C13": CEC2017_C13,
        "C14": CEC2017_C14,
        "C15": CEC2017_C15,
        "C16": CEC2017_C16,
        "C17": CEC2017_C17,
        "C18": CEC2017_C18,
        "C19": CEC2017_C19,
        # "C20": CEC2017_C20,
        # "C21": CEC2017_C21,
        # "C22": CEC2017_C22,
        # "C23": CEC2017_C23,
        # "C24": CEC2017_C24,
        # "C25": CEC2017_C25,
        # "C27": CEC2017_C27,
        # "C28": CEC2017_C28
    }

    bchms = {
        "adaptive_centroid": BCHM.adaptive_centroid,
        "centroid_repair": BCHM.centroid_repair,
        "ADS": BCHM.ADS,
        "beta": BCHM.beta,
        "boundary": BCHM.boundary,
        "centroid": BCHM.centroid,
        "reflection": BCHM.reflection,
        "random": BCHM.random_component,
        "res&rand": None,
        "evolutionary": BCHM.evolutionary,
        "wrapping": BCHM.wrapping,
        "vector_wise_correction": BCHM.vector_wise_correction,
    }

    DIRECTORY = "mnt/data/cec2017"
    PROBLEM_PREFIX = "CEC2017"

    main(problems, bchms, DIRECTORY, PROBLEM_PREFIX)
    # plot_fitness_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, "C07", exclude=["boundary"])
    # plot_violations_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, "C02")
