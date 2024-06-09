import numpy as np
import ast

def calculate_mean_boxes_constraints(*results) -> tuple:
    try:
        max_lengths = [max(len(seq) for seq in result) for result in results]
        max_length = max(max_lengths)

        normalized_results = [np.array([seq + [np.nan] * (max_length - len(seq)) for seq in result]) for result in results]

        mean_results = [np.nanmean(normalized_result, axis=0) for normalized_result in normalized_results]

        return tuple(mean_results)
    except RuntimeWarning:
        pass
    
def save_mean_to_text_file(mean, method_name:str, filename:str="report/mean-text-file/means.txt"):
    try:
        with open(filename, 'a') as archivo: 
            archivo.write(f'{method_name}: {mean}\n')
        print(f'Promedio {mean} calculado por {mean} guardado en {filename} exitosamente.')
    except Exception as e:
        print(f'Ocurrió un error al guardar el promedio: {e}')
        
def get_mean_for_method_name(filename="report/mean-text-file/means.txt", method_specified=''):
    try:
        with open(filename, 'r') as archivo:
            lineas = archivo.readlines()
            for linea in lineas:
                # Verificar si la línea contiene el método especificado
                if method_specified in linea:
                    parts = linea.split(': (array(')
                    if len(parts) == 2 and method_specified.strip() == parts[0].strip():
                        array_str = parts[1].rstrip('),)\n')
                        array = [ast.literal_eval(array_str)]
                        return tuple(array)
        print(f'No se encontró el método {method_specified} en el archivo.')
        return None
    except Exception as e:
        print(f'Ocurrió un error al leer el archivo: {e}')
        return None
