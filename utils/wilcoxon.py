import pandas as pd
from scipy.stats import wilcoxon

def compare_proposals_from_csv(b, p, file_path_proposal, file_path_baseline, column_name):
    """
    Compara dos listas de valores en dos archivos CSV diferentes utilizando la prueba de Wilcoxon.
    
    Parameters:
    file_path_proposal (str): Ruta al archivo CSV de la propuesta.
    file_path_baseline (str): Ruta al archivo CSV del estándar de comparación.
    column_name (str): Nombre de la columna de "Fitness" en ambos archivos.
    
    Returns:
    str: Resultado de la comparación, incluyendo los arrays, medias y valor p.
    """
    # Leer los archivos CSV
    df_proposal = pd.read_csv(file_path_proposal)
    df_baseline = pd.read_csv(file_path_baseline)
    
    # Extraer los datos de la columna especificada
    proposal = df_proposal[column_name].dropna().values
    baseline = df_baseline[column_name].dropna().values
    
    if len(proposal) != len(baseline):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    # Calcular las medias
    mean_proposal = proposal.mean()
    mean_baseline = baseline.mean()
    
    # Aplicar la prueba de Wilcoxon unilateral (menor es mejor)
    stat, p_value = wilcoxon(baseline, proposal, alternative='less')
    
    # Determinar el resultado
    alpha = 0.05
    if p_value < alpha:
        result = "La diferencia es estadísticamente significativa."
        if mean_proposal < mean_baseline:
            result += f" (gana mean_proposal {p})."
        else:
            result += f" (gana mean_baseline {b})."
    else:
        result = "No hay una diferencia estadísticamente significativa (empate)."
    
    # Imprimir arrays, medias, valor p y resultado
    output = ( 
              f"Media de la propuesta: {mean_proposal}\n"
              f"Media de la línea base: {mean_baseline}\n"
              f"Valor p: {p_value}\n"
              f"Valor stat: {stat}\n"
              f"Resultado: {result}")
    
    return output
