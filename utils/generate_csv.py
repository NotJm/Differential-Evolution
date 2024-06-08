import csv
import os

# Función para guardar los resultados óptimos en un CSV, agregando nuevas filas
def save_results_csv_file(datos: dict, filename: str):
    # Verificar si el archivo ya existe
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=datos.keys())
        # Escribir los encabezados solo si el archivo no existe
        if not file_exists:
            writer.writeheader()
        # Escribir los resultados
        writer.writerow(datos)
