import pandas as pd
import os
from utils.utility import ensure_directory_exists

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
    
def save_results_to_csv(results, directory, problem_prefix):
    df = pd.DataFrame(results)
    ensure_directory_exists(directory)
    filepath = f"{directory}/{problem_prefix}.csv"
    df.to_csv(filepath, index=False)

def adaptive_results(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    
    pivot_df = df.pivot_table(index='Problema', columns='Restricción', values=['Fitness Promedio', 'Ejecuciones Factibles'], aggfunc='first')
    
    formatted_df = pd.DataFrame(index=pivot_df.index)
    
    for col in pivot_df.columns.levels[1]:
        fitness_col = ('Fitness Promedio', col)
        feasible_col = ('Ejecuciones Factibles', col)
        fitness_vals = pivot_df[fitness_col].fillna('-').astype(str)
        
        feasible_vals = pivot_df[feasible_col].apply(lambda x: '-' if pd.isna(x) else str(int(x)))
        
        formatted_df[col] = fitness_vals + '(' + feasible_vals + ')'

    
    
    formatted_df.to_csv(output_filename, encoding='latin1')
    
def adaptive_results_violations(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    
    pivot_df = df.pivot_table(index='Problema', columns='Restricción', values=['Violations Promedio'], aggfunc='first')
    
    formatted_df = pd.DataFrame(index=pivot_df.index)
    
    for col in pivot_df.columns.levels[1]:
        fitness_col = ('Violations Promedio', col)
        formatted_df[col] = pivot_df[fitness_col].astype(str)

    
    
    formatted_df.to_csv(output_filename, encoding='latin1')
    
def generate_summary(base_dir, output_path):
    final_data = []

    for restriction in os.listdir(base_dir):
        restriction_dir = os.path.join(base_dir, restriction)
        if os.path.isdir(restriction_dir):
            for file in os.listdir(restriction_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restriction_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Verifica que las columnas necesarias estén en el DataFrame
                        if 'Fitness' in df.columns and 'Violations' in df.columns:
                            feasible_df = df[df['Violations'] == 0]
                            
                            avg_fitness = feasible_df['Fitness'].mean()
                            
                            feasible_executions = feasible_df.shape[0]
                            
                            problem = file.split('_')[1]
                            
                            final_data.append([problem, restriction, avg_fitness, feasible_executions])
                        else:
                            print(f"Skipping {file_path} as it does not contain required columns.")
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    final_df = pd.DataFrame(final_data, columns=['Problema', 'Restricción', 'Fitness Promedio', 'Ejecuciones Factibles'])

    final_df = final_df.sort_values(by=['Problema'])
    
    final_df.to_csv(output_path, index=False)
    
def generate_summary_for_constraint(base_dir, output_path, constriants):
    final_data = []

    for restriction in os.listdir(base_dir):
        restriction_dir = os.path.join(base_dir, restriction)
        if os.path.isdir(restriction_dir):
            for file in os.listdir(restriction_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restriction_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Verifica que las columnas necesarias estén en el DataFrame
                        if 'Fitness' in df.columns and 'Violations' in df.columns:
                            feasible_df = df[df['Violations'] == 0]
                            
                            avg_fitness = feasible_df['Fitness'].mean()
                            
                            feasible_executions = feasible_df.shape[0]
                            
                            constriant = file.split('_')[2]
                            
                            problem = file.split('_')[1]
                            
                            # Procesa solo los problemas especificados en la lista
                            if constriants and constriant not in constriants:
                                continue
                            
                            final_data.append([problem, restriction, avg_fitness, feasible_executions])
                        else:
                            print(f"Skipping {file_path} as it does not contain required columns.")
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    final_df = pd.DataFrame(final_data, columns=['Problema', 'Restricción', 'Fitness Promedio', 'Ejecuciones Factibles'])
    
    final_df = final_df.sort_values(by=['Problema'])
    
    final_df.to_csv(output_path, index=False)
    
def generate_summary_for_problem(base_dir, output_path, problems=[]):
    final_data = []

    for restriction in os.listdir(base_dir):
        restriction_dir = os.path.join(base_dir, restriction)
        if os.path.isdir(restriction_dir):
            for file in os.listdir(restriction_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restriction_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Verifica que las columnas necesarias estén en el DataFrame
                        if 'Fitness' in df.columns and 'Violations' in df.columns:
                            feasible_df = df[df['Violations'] == 0]
                            
                            avg_fitness = feasible_df['Fitness'].mean()
                            
                            feasible_executions = feasible_df.shape[0]
                            
                            problem = file.split('_')[1]
                            
                            # Procesa solo los problemas especificados en la lista
                            if problems and problem not in problems:
                                continue
                            
                            final_data.append([problem, restriction, avg_fitness, feasible_executions])
                        else:
                            print(f"Skipping {file_path} as it does not contain required columns.")
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    final_df = pd.DataFrame(final_data, columns=['Problema', 'Restricción', 'Fitness Promedio', 'Ejecuciones Factibles'])
    
    final_df = final_df.sort_values(by=['Problema'])
    
    final_df.to_csv(output_path, index=False)

def generate_summary_violations(base_dir, output_path, problems=[]):
    final_data = []

    for restriction in os.listdir(base_dir):
        restriction_dir = os.path.join(base_dir, restriction)
        if os.path.isdir(restriction_dir):
            for file in os.listdir(restriction_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restriction_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Verifica que las columnas necesarias estén en el DataFrame
                        if 'Violations' in df.columns:
                            
                            avg_violations = df['Violations'].mean()
                            
                            
                            problem = file.split('_')[1]
                            
                            # Procesa solo los problemas especificados en la lista
                            if problems and problem not in problems:
                                continue
                            
                            final_data.append([problem, restriction, avg_violations])
                        else:
                            print(f"Skipping {file_path} as it does not contain required columns.")
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    final_df = pd.DataFrame(final_data, columns=['Problema', 'Restricción', 'Violations Promedio'])
    
    final_df = final_df.sort_values(by=['Problema'])
    
    final_df.to_csv(output_path, index=False)    

def generate_summary_for_constraint(base_dir, output_path, constraint):
    final_data = []

    for restriction in os.listdir(base_dir):
        # Procesa solo los problemas especificados en la lista de constraints
        if restriction != constraint:
            continue
        
        restriction_dir = os.path.join(base_dir, restriction)
        if os.path.isdir(restriction_dir):
            for file in os.listdir(restriction_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restriction_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Verifica que las columnas necesarias estén en el DataFrame
                        if 'Fitness' in df.columns and 'Violations' in df.columns:
                            feasible_df = df[df['Violations'] == 0]
                            
                            avg_fitness = feasible_df['Fitness'].mean()
                            
                            feasible_executions = feasible_df.shape[0]
                            
                            problem = file.split('_')[1]
                            
                            final_data.append([problem, restriction, avg_fitness, feasible_executions])
                        else:
                            print(f"Skipping {file_path} as it does not contain required columns.")
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    final_df = pd.DataFrame(final_data, columns=['Problema', 'Restricción', 'Fitness Promedio', 'Ejecuciones Factibles'])
    
    final_df = final_df.sort_values(by=['Problema'])
    
    final_df.to_csv(output_path, index=False)