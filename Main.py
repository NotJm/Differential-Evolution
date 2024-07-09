import pandas as pd
import smop.main
from core.bchms import BCHM
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

    DIRECTORY = "mnt/data/cec2017"
    PROBLEM_PREFIX = "CEC2017"

    problems = {
        "C01": CEC2017_C01,
        "C02": CEC2017_C02,
        "C03": CEC2017_C03,
        "C04": CEC2017_C04,
        "C05": CEC2017_C05,
    }

    # 5, 8, 17

    bchms = {
        "TPC": BCHM.evo_cen_beta,
        # "centroid": BCHM.centroid,
        # "res&rand": None,
        # "beta": BCHM.beta,
        # "boundary": BCHM.boundary,
        # "reflection": BCHM.reflection,
        # "random": BCHM.random_component,
        # "evolutionary": BCHM.evolutionary,
        # "wrapping": BCHM.wrapping,
        # "vector_wise_correction": BCHM.vector_wise_correction,
    }

    exclude = [
        # "boundary",
        # "beta",
        # "centroid",
        # "vector_wise_correction",
        # "reflection",
        # "evolutionary",
        # "wrapping",
        # "random",
        # "res&rand",
    ]

    main(problems, bchms, DIRECTORY, PROBLEM_PREFIX)

    # plot_fitness_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, "C22", exclude)
    # plot_violations_boxplot_from_csvs(DIRECTORY, PROBLEM_PREFIX, "C17", exclude)

    # generate_summary(DIRECTORY, "CEC2017.csv")
    # generate_summary_violations(
    #     DIRECTORY, "CEC2017violations.csv", 
    #     ["C06",
    #      "C07",
    #      "C08",
    #      "C09",
    #      "C10",
    #      "C11",
    #      "C12",
    #      "C14",
    #      "C15",
    #      "C16",
    #      "C17",
    #      "C18",
    #      "C23",
    #      "C24",
    #     ]
    # )

    # # # adaptive_results(f"CEC2017.csv", "prev2CEC2017.csv")
    # adaptive_results_violations("CEC2017violations.csv", "prevCEC2017violations.csv")
    
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Valores de los límites inferiores y superiores
    # lower = np.array([0, 0])
    # upper = np.array([5, 5])

    # # Coordenadas de los puntos relevantes
    # x = np.array([-1, 6])  # Solución inicial fuera de los límites
    # best_solution = np.array([2, 3])
    # x_evo = np.array([0.8, 3.6])  # Solución ajustada
    # x_cen = np.array([0.9, 2.8])  # Solución centrada

    # # Crear la figura y los ejes
    # fig, ax = plt.subplots()

    # # Dibujar el espacio de búsqueda
    # rect = plt.Rectangle(lower, upper[0] - lower[0], upper[1] - lower[1], edgecolor='black', facecolor='none')
    # ax.add_patch(rect)

    # # Dibujar los puntos relevantes
    # ax.plot(x[0], x[1], 'ro', label='x (Inicial)')
    # ax.plot(x_evo[0], x_evo[1], 'bo', label='x_evo')
    # ax.plot(x_cen[0], x_cen[1], 'go', label='x_cen')

    # # Anotaciones para los puntos
    # ax.annotate('x (Inicial)', xy=(x[0], x[1]), xytext=(x[0]-1.5, x[1]+0.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    # ax.annotate('x_evo', xy=(x_evo[0], x_evo[1]), xytext=(x_evo[0]+0.2, x_evo[1]-1),
    #             arrowprops=dict(facecolor='blue', shrink=0.05))
    # ax.annotate('x_cen', xy=(x_cen[0], x_cen[1]), xytext=(x_cen[0]-1, x_cen[1]-0.5),
    #             arrowprops=dict(facecolor='green', shrink=0.05))

    # # Configuración del gráfico
    # ax.set_xlim(-2, 7)
    # ax.set_ylim(-2, 7)
    # ax.set_xlabel('Dimensión 1')
    # ax.set_ylabel('Dimensión 2')
    # ax.set_title('Espacio de Búsqueda y Corrección del Individuo')
    # ax.legend()
    # plt.grid(True)
    # plt.show()

