import pandas as pd
from utils.utility import ensure_directory_exists

# ! Save Data
def save_results_to_csv_constraint(
    problem_name,
    constraint_name,
    fitness_data,
    violations_data,
    directory,
    problem_prefix,
):
    directory = f"{directory}/{constraint_name}"
    ensure_directory_exists(directory)
    df = pd.DataFrame({"Fitness": fitness_data, "Violations": violations_data})
    filename = f"{directory}/{problem_prefix}_{problem_name}_{constraint_name}.csv"
    df.to_csv(filename, index=False)


# ! Save
def save_results_to_csv(results, directory, problem_prefix):
    df = pd.DataFrame(results)
    ensure_directory_exists(directory)
    filepath = f"{directory}/{problem_prefix}.csv"
    df.to_csv(filepath, index=False)