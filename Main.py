import pandas as pd
from functions.cec2020problems import *
from functions.cec2006problems import *
from functions.cec2017problems_update import *
from functions.cec2010problems_update import *
from utils.execute import execute_algorithm
from utils.plotting import (
    plot_fitness_boxplot_all,
    plot_violations_boxplot_all,
    plot_fitness_boxplot_from_csvs,
    plot_violations_boxplot_from_csvs,
)
from utils.save_file import (
    save_results_to_csv,
    adaptive_results,
    adaptive_results_violations,
    generate_summary,
    generate_summary_for_problem,
    generate_summary_violations,
    generate_summary_for_constraint,
)


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

    results_df = pd.DataFrame(
        results,
        columns=[
            "Problema",
            "Restricción",
            "Fitness Promedio",
            "Ejecuciones Factibles",
        ],
    )

    # save_results_to_csv(results_df, directory, problem_prefix)


if __name__ == "__main__":

    DIRECTORY = "mnt/data/cec2017"
    PROBLEM_PREFIX = "CEC2017"

    PROBLEMS = {
        "CEC2017-1": CEC2017_C01,
        "CEC2017-2": CEC2017_C02,
        "CEC2017-3": CEC2017_C03,
        "CEC2017-4": CEC2017_C04,
        "CEC2017-5": CEC2017_C05,
        "CEC2017-6": CEC2017_C06,
        "CEC2017-7": CEC2017_C07,
        "CEC2017-8": CEC2017_C08,
        "CEC2017-9": CEC2017_C09,
        "CEC2017-10": CEC2017_C10,
        "CEC2017-11": CEC2017_C11,
        "CEC2017-12": CEC2017_C12,
        "CEC2017-13": CEC2017_C13,
        "CEC2017-14": CEC2017_C14,
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
    import csv
    with open('Experiment.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['BCHM', 'Diversidad', 'Porcentaje de Factibilidad', 'Porcentaje Evaluaciones Realizadas', 'Mejor', 'Problema'])

    BCHM = {"dataset": None}

    # EXCLUDE = ["evo&cen", "TPC"]    

    # CURRENT_PROBLEM = "C13"

    main(PROBLEMS, BCHM, DIRECTORY, PROBLEM_PREFIX)

    # plot_fitness_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, CURRENT_PROBLEM, EXCLUDE)

    # plot_fitness_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, "C08", exclude)

    # generate_summary_for_constraint(DIRECTORY, "CEC2017MulyWeight", "MultyWeight")
