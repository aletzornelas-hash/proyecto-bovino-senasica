import csv
import time


def cargar_datos_ram(ruta_archivo):
    inicio = time.perf_counter_ns()

    lista_ganado = []
    with open(ruta_archivo, newline='', encoding='utf-8-sig') as f:
        for fila in csv.DictReader(f):
            fila['siniiga_id'] = int(fila['siniiga_id'])
            fila['peso_kg'] = int(fila['peso_kg'])
            fila['fecha_vacunacion'] = int(fila['fecha_vacunacion'])
            fila['edad_meses'] = int(fila['edad_meses'])
            lista_ganado.append(fila)

    tiempo_carga_ns = time.perf_counter_ns() - inicio
    return lista_ganado, tiempo_carga_ns