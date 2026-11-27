"""
MÉTODOS DE ORDENACIÓN Y BÚSQUEDA
Proyecto 2: Rastreo e Inventario Bovino para Exportación y Control Zoosanitario

Ejecutar con:  streamlit run app.py
Archivos necesarios en la misma carpeta: app.py, datos.py, contexto.txt y ganado_15000.csv / ganado_30000.csv
"""

import csv
from datetime import datetime
import hashlib
import hmac
import html
import io
import json
import os
from pathlib import Path
import random
import secrets
import sys
import time
import unicodedata

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from datos import cargar_datos_ram  # noqa: E402

# ----------------------------------------------------------------------------
# CONSTANTES Y ARCHIVOS
# ----------------------------------------------------------------------------
DATASETS = [
    ("ganado_15000.csv", "15,000"),
    ("ganado_30000.csv", "30,000"),
]
TAMANOS_PERMITIDOS = {15000, 30000}
OPCIONES_MUESTRA = [10, 20, 30, 40, 50, 60]

BITACORA_ORDENACION = "bitacora_ordenacion.txt"
BITACORA_BUSQUEDA = "bitacora_busqueda.txt"
BITACORA_GLOBAL = "registro_global_visitas.txt"
ARCHIVO_ALFA_GLOBAL = "alfa_global.json"
ARCHIVO_USUARIOS = "usuarios.json"
ARCHIVO_CONTEXTO = "contexto.txt"

COLUMNAS = [
    "siniiga_id",
    "peso_kg",
    "fecha_vacunacion",
    "raza_bovino",
    "estado_origen",
    "municipio_rancho",
    "estatus_exportacion",
    "edad_meses",
    "sexo_bovino",
    "nombre_rancho",
]
CAMPOS_NUMERICOS = {"siniiga_id", "peso_kg", "fecha_vacunacion", "edad_meses"}
CAMPOS_CLAVE = {1: "siniiga_id", 2: "peso_kg", 3: "fecha_vacunacion"}
ETIQUETA = {c: (c + "*" if c in CAMPOS_CLAVE.values() else c) for c in COLUMNAS}

LIMITE_RANGO_CONTEO = 2_000_000


def fmt(n):
    return f"{n:,}"


def archivo_ordenado():
    return st.session_state.get("archivo_ordenado") or "ganado_N_ordenado.csv"


# ----------------------------------------------------------------------------
# VALIDACIÓN Y CARGA MANUAL DE CSV EN RAM
# ----------------------------------------------------------------------------
def cargar_csv_manual_en_ram(archivo_subido):
    """
    Valida estrictamente un CSV subido manualmente:
    1. Extensión .csv y codificación UTF-8 / Latin-1 legible.
    2. Exactamente las 10 columnas requeridas.
    3. Tipos enteros válidos en los campos numéricos.
    4. Exactamente 15,000 o 30,000 registros.
    Retorna: (lista_diccionarios, tiempo_ns) o lanza ValueError.
    """
    nombre = archivo_subido.name or ""
    if not nombre.lower().endswith(".csv"):
        raise ValueError(
            f"Archivo inválido ('{nombre}'): solo se permiten archivos con extensión .csv."
        )

    crudo = archivo_subido.getvalue()
    if not crudo:
        raise ValueError("El archivo está vacío.")

    try:
        texto = crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            texto = crudo.decode("latin-1")
        except Exception:
            raise ValueError(
                "No se pudo leer el archivo como texto CSV válido (codificación incompatible)."
            )

    inicio = time.perf_counter_ns()
    lector = csv.DictReader(io.StringIO(texto))
    encabezados = lector.fieldnames or []

    if set(encabezados) != set(COLUMNAS) or len(encabezados) != len(COLUMNAS):
        faltantes = [c for c in COLUMNAS if c not in encabezados]
        extra =
