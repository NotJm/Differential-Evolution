import os
import random
import string
import glob
import pandas as pd

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def generate_random_filename():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

def rename_problem_prefix(directory, old_prefix, new_prefix, extensions=[".csv"], problems=[]):
    renamed_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if any(filename.endswith(ext) for ext in extensions):
                if filename.startswith(old_prefix):
                    for problem in problems:
                        if problem in filename:
                            new_filename = filename.replace(old_prefix, new_prefix, 1)
                            old_filepath = os.path.join(root, filename)
                            new_filepath = os.path.join(root, new_filename)
                            os.rename(old_filepath, new_filepath)
                            renamed_files.append((old_filepath, new_filepath))
                            break
    return renamed_files

def calculate_average_violations(root_folder, problem_prefix):
    average_violations = {}

    # Iterar sobre todas las restricciones (subcarpetas en root_folder)
    for restriction_folder in glob.glob(os.path.join(root_folder, "*")):
        if os.path.isdir(restriction_folder):
            restriction_name = os.path.basename(restriction_folder)
            violation_counts = []

            # Iterar sobre todos los archivos CSV dentro de la carpeta de restricción
            for csv_file in glob.glob(
                os.path.join(restriction_folder, f"{problem_prefix}*.csv")
            ):
                df = pd.read_csv(csv_file)
                # Obtener las infracciones y agregarlas a la lista
                violation_counts.extend(df["Violations"])

            if violation_counts:
                # Calcular el promedio de infracciones
                average_violations[restriction_name] = sum(violation_counts) / len(
                    violation_counts
                )
            else:
                average_violations[restriction_name] = 0

    return average_violations

def calculate_mean_convergence_violations(bounds, all_convergence_data):
    mean_convergence_violations = {}
    for constraint_name in bounds:
        if constraint_name in all_convergence_data:
            mean_convergence_violations[constraint_name] = [
                sum(gen) / len(gen)
                for gen in zip(*all_convergence_data[constraint_name])
        ]
    return mean_convergence_violations


