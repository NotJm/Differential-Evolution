import pandas as pd
from utils.constants import *
from utils.wilcoxon import compare_proposals_from_csv
from utils.execute import execute_algorithm
from utils.utility import calculate_mean_convergence_violations
from utils.plotting import (
    plot_fitness_boxplot_from_csvs,
    plot_violations_boxplot_from_csvs,
    plot_convergence_from_json,
)
from utils.save_file import (
    save_results_to_csv,
    adaptive_results,
    adaptive_results_violations,
    generate_summary,
    generate_summary_for_problem,
    generate_summary_violations,
    generate_summary_for_constraint,
    save_results_to_json,
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

            data = calculate_mean_convergence_violations(bchms, all_convergence_data)
            save_results_to_json(data, problem_name, directory, problem_prefix, bchm)

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

    # main(PROBLEMS, BCHM, DIRECTORY, PROBLEM_PREFIX)

    plot_convergence_from_json(CURRENT_PROBLEM, DIRECTORY, PROBLEM_PREFIX)

    # plot_fitness_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, CURRENT_PROBLEM, [])

    # plot_violations_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, CURRENT_PROBLEM, EXCLUDE)

    # generate_summary_for_problem(DIRECTORY, "CEC2024.csv", ["C19", "C20"])

    # adaptive_results("CEC2024.csv", "CEC2024.csv")

    # generate_summary_violations(
    #     DIRECTORY,
    #     f"Informe de Violaciones {PROBLEM_PREFIX}.csv",
    #     [
    #         "C15",
    #         "C16",
    #         "C17",
    #         "C18",
    #         "C21",
    #         "C22",
    #         "C23",
    #         "C24",
    #         "C25",
    #         "C26",
    #         "C27",
    #         "C28",
    #     ],
    # )

    # adaptive_results_violations(
    #     f"Informe de Violaciones {PROBLEM_PREFIX}.csv",
    #     f"Informe de Violaciones {PROBLEM_PREFIX}.csv",
    # )

    # for p in PROPOSAL:
    #     print(CURRENT_PROBLEM)
    #     print(p)
    #     DIRECTORY_PROPOSAL = f"{DIRECTORY}/{p}/{PROBLEM_PREFIX}_{CURRENT_PROBLEM}_{p}.csv"
    #     result = compare_proposals_from_csv(BASELINE, p, DIRECTORY_PROPOSAL, DIRECTORY_BASELINE, "Violations")
    #     print(result)
    