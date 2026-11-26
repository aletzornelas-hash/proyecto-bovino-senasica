"""
MÉTODOS DE ORDENACIÓN Y BÚSQUEDA
Proyecto 2: Rastreo e Inventario Bovino para Exportación y Control Zoosanitario

Ejecutar con:  streamlit run app.py
Archivos necesarios en la misma carpeta: app.py, datos.py, contexto.txt y ganado_15000.csv y/o ganado_30000.csv
"""

import csv
from datetime import datetime
import hashlib
import hmac
import html
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
# Bases de datos disponibles: (archivo, etiqueta del botón)
DATASETS = [
    ("ganado_15000.csv", "15,000"),
    ("ganado_30000.csv", "30,000"),
]
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


MUESTRA_REGISTROS = 60  # primeros 60 y últimos 60, sin importar la base cargada


def tamano_muestra(n):
    """Siempre 60 + 60 (con menos de 60 registros se muestran todos)."""
    return min(MUESTRA_REGISTROS, n)


def archivo_ordenado():
    """Nombre del CSV ordenado según la cantidad real de registros cargados (len(datos))."""
    return st.session_state.get("archivo_ordenado") or "ganado_N_ordenado.csv"


# ----------------------------------------------------------------------------
# ALGORITMOS DE ORDENACIÓN (desde cero, sin sort()/sorted())
# ----------------------------------------------------------------------------
def _rango(datos, campo):
    minimo = maximo = datos[0][campo]
    for reg in datos:
        v = reg[campo]
        if v < minimo:
            minimo = v
        if v > maximo:
            maximo = v
    return minimo, maximo


def radix_sort(datos, campo="siniiga_id"):
    """Método 1: Radix Sort LSD en base 10."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    n = len(datos)
    if n < 2:
        return list(datos), time.perf_counter_ns() - inicio, 0, 0

    minimo, maximo = _rango(datos, campo)
    amplitud = maximo - minimo
    origen = list(datos)
    exp = 1
    while amplitud // exp > 0:
        conteo = [0] * 10
        for reg in origen:
            conteo[((reg[campo] - minimo) // exp) % 10] += 1
        for d in range(1, 10):
            conteo[d] += conteo[d - 1]
        destino = [None] * n
        for i in range(n - 1, -1, -1):
            reg = origen[i]
            d = ((reg[campo] - minimo) // exp) % 10
            conteo[d] -= 1
            destino[conteo[d]] = reg
            movimientos += 1
        origen = destino
        exp *= 10
    return origen, time.perf_counter_ns() - inicio, comparaciones, movimientos


def counting_sort(datos, campo="siniiga_id"):
    """Método 2: Counting Sort estable."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    n = len(datos)
    if n < 2:
        return list(datos), time.perf_counter_ns() - inicio, 0, 0

    minimo, maximo = _rango(datos, campo)
    k = maximo - minimo + 1
    if k > LIMITE_RANGO_CONTEO:
        raise ValueError("Rango demasiado grande para Counting Sort.")
    conteo = [0] * k
    for reg in datos:
        conteo[reg[campo] - minimo] += 1
    for i in range(1, k):
        conteo[i] += conteo[i - 1]
    salida = [None] * n
    for i in range(n - 1, -1, -1):
        reg = datos[i]
        idx = reg[campo] - minimo
        conteo[idx] -= 1
        salida[conteo[idx]] = reg
        movimientos += 1
    return salida, time.perf_counter_ns() - inicio, comparaciones, movimientos


def bucket_sort(datos, campo="peso_kg"):
    """Método 3: Bucket Sort con Insertion Sort estable dentro de cada cubeta."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    n = len(datos)
    if n < 2:
        return list(datos), time.perf_counter_ns() - inicio, 0, 0

    minimo, maximo = _rango(datos, campo)
    rango = maximo - minimo + 1
    num_cubetas = min(n, rango)
    cubetas = [[] for _ in range(num_cubetas)]
    for reg in datos:
        cubetas[(reg[campo] - minimo) * num_cubetas // rango].append(reg)
        movimientos += 1

    resultado = []
    for cubeta in cubetas:
        for i in range(1, len(cubeta)):
            actual = cubeta[i]
            clave = actual[campo]
            j = i - 1
            while j >= 0:
                comparaciones += 1
                if cubeta[j][campo] > clave:
                    cubeta[j + 1] = cubeta[j]
                    movimientos += 1
                    j -= 1
                else:
                    break
            cubeta[j + 1] = actual
            movimientos += 1
        for reg in cubeta:
            resultado.append(reg)
            movimientos += 1
    return (
        resultado,
        time.perf_counter_ns() - inicio,
        comparaciones,
        movimientos,
    )


def quick_sort_iterativo(datos, campo="fecha_vacunacion"):
    """Método 4: Quick Sort iterativo con partición de 3 vías."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    lista = list(datos)
    n = len(lista)
    pila = [(0, n - 1)]
    while pila:
        lo, hi = pila.pop()
        if lo >= hi:
            continue
        a = lista[random.randint(lo, hi)][campo]
        b = lista[random.randint(lo, hi)][campo]
        d = lista[random.randint(lo, hi)][campo]
        comparaciones += 1
        if a < b:
            comparaciones += 1
            if b < d:
                pivote = b
            else:
                comparaciones += 1
                pivote = d if a < d else a
        else:
            comparaciones += 1
            if a < d:
                pivote = a
            else:
                comparaciones += 1
                pivote = d if b < d else b

        lt, i, gt = lo, lo, hi
        while i <= gt:
            v = lista[i][campo]
            comparaciones += 1
            if v < pivote:
                lista[lt], lista[i] = lista[i], lista[lt]
                movimientos += 2
                lt += 1
                i += 1
            else:
                comparaciones += 1
                if v > pivote:
                    lista[i], lista[gt] = lista[gt], lista[i]
                    movimientos += 2
                    gt -= 1
                else:
                    i += 1
        izq, der = (lo, lt - 1), (gt + 1, hi)
        if (izq[1] - izq[0]) > (der[1] - der[0]):
            pila.append(izq)
            pila.append(der)
        else:
            pila.append(der)
            pila.append(izq)
    return lista, time.perf_counter_ns() - inicio, comparaciones, movimientos


def merge_sort(datos, campo="fecha_vacunacion"):
    """Método 5: Merge Sort iterativo (bottom-up), estable."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    n = len(datos)
    src = list(datos)
    dst = [None] * n
    ancho = 1
    while ancho < n:
        for lo in range(0, n, 2 * ancho):
            mid = min(lo + ancho, n)
            hi = min(lo + 2 * ancho, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                comparaciones += 1
                if src[j][campo] < src[i][campo]:
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
                movimientos += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
                movimientos += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
                movimientos += 1
        src, dst = dst, src
        ancho *= 2
    return src, time.perf_counter_ns() - inicio, comparaciones, movimientos


ALGORITMOS = {
    1: ("Radix Sort", radix_sort),
    2: ("Bucket Sort", bucket_sort),
    3: ("Quick Sort", quick_sort_iterativo),
}
CLAVE_DE_CAMPO = {campo: clave for clave, campo in CAMPOS_CLAVE.items()}
ICONOS = {
    "siniiga_id": "🆔",
    "peso_kg": "⚖️",
    "fecha_vacunacion": "💉",
    "raza_bovino": "🐂",
    "estado_origen": "🗺️",
    "municipio_rancho": "📍",
    "estatus_exportacion": "🚢",
    "edad_meses": "🎂",
    "sexo_bovino": "⚥",
    "nombre_rancho": "🏡",
}


# ----------------------------------------------------------------------------
# ALGORITMOS DE BÚSQUEDA
# ----------------------------------------------------------------------------
def busqueda_binaria(datos_ordenados, campo, valor_buscado):
    inicio = time.perf_counter_ns()
    comparaciones = 0
    n = len(datos_ordenados)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        comparaciones += 1
        if datos_ordenados[mid][campo] < valor_buscado:
            lo = mid + 1
        else:
            hi = mid
    encontrado = False
    if lo < n:
        comparaciones += 1
        encontrado = datos_ordenados[lo][campo] == valor_buscado
    tiempo = time.perf_counter_ns() - inicio
    if not encontrado:
        return None, None, [], tiempo, comparaciones

    todos = []
    i = lo
    while i < n and datos_ordenados[i][campo] == valor_buscado:
        todos.append((i + 1, datos_ordenados[i]))
        i += 1
    return lo + 1, datos_ordenados[lo], todos, tiempo, comparaciones


def normalizar_texto(texto):
    sin_acentos = "".join(
        c
        for c in unicodedata.normalize("NFKD", str(texto))
        if not unicodedata.combining(c)
    )
    return " ".join(sin_acentos.lower().strip().split())


def busqueda_lineal(datos, campo, valor_buscado):
    inicio = time.perf_counter_ns()
    comparaciones = 0
    indice = None
    es_texto = campo not in CAMPOS_NUMERICOS
    if es_texto:
        objetivo = normalizar_texto(valor_buscado)
        if not objetivo:
            return None, None, [], time.perf_counter_ns() - inicio, 0
        primer_parcial = None
        for i, reg in enumerate(datos):
            comparaciones += 1
            v = normalizar_texto(reg[campo])
            if v == objetivo:
                indice = i
                break
            if primer_parcial is None and objetivo in v:
                primer_parcial = i
        exacta = indice is not None
        if not exacta:
            indice = primer_parcial
    else:
        for i, reg in enumerate(datos):
            comparaciones += 1
            if reg[campo] == valor_buscado:
                indice = i
                break
    tiempo = time.perf_counter_ns() - inicio
    if indice is None:
        return None, None, [], tiempo, comparaciones

    todos = []
    if es_texto:
        for i, reg in enumerate(datos):
            v = normalizar_texto(reg[campo])
            if (v == objetivo) if exacta else (objetivo in v):
                todos.append((i + 1, reg))
    else:
        for i, reg in enumerate(datos):
            if reg[campo] == valor_buscado:
                todos.append((i + 1, reg))
    return indice + 1, datos[indice], todos, tiempo, comparaciones


# ----------------------------------------------------------------------------
# PERSISTENCIA GLOBAL Y BITÁCORAS
# ----------------------------------------------------------------------------
def leer_alfa_global():
    ruta = BASE_DIR / ARCHIVO_ALFA_GLOBAL
    base = {nombre: 0 for nombre, _ in ALGORITMOS.values()}
    if ruta.exists():
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            for k in base:
                base[k] = int(datos.get(k, 0))
        except Exception:
            pass
    return base


def incrementar_alfa_global(nombre_metodo):
    conteo = leer_alfa_global()
    conteo[nombre_metodo] = conteo.get(nombre_metodo, 0) + 1
    ruta = BASE_DIR / ARCHIVO_ALFA_GLOBAL
    ruta.write_text(
        json.dumps(conteo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return conteo


def registrar_evento(tipo_accion, detalle, tiempo_ns):
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    usuario = st.session_state.get("usuario_activo") or "Anónimo"
    evento = {
        "Fecha/Hora": ahora,
        "Operador": usuario,
        "Acción": tipo_accion,
        "Detalle": detalle,
        "Tiempo (ns)": fmt(tiempo_ns),
    }
    st.session_state.historial_personal.insert(0, evento)

    linea_global = f"[{ahora}] | Operador: {usuario} | {tipo_accion} | {detalle} | {fmt(tiempo_ns)} ns\n"
    with open(BASE_DIR / BITACORA_GLOBAL, "a", encoding="utf-8") as f:
        f.write(linea_global)


def escribir_bitacora(nombre_archivo, texto):
    usuario = st.session_state.get("usuario_activo") or "Anónimo"
    with open(BASE_DIR / nombre_archivo, "a", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(
            f"Fecha y hora: {datetime.now():%Y-%m-%d %H:%M:%S} | Operador: {usuario} | "
            f"Dataset: {st.session_state.get('dataset_activo') or 'N/D'}\n"
        )
        f.write(texto.strip("\n") + "\n\n")


def guardar_csv_ordenado(registros):
    with open(
        BASE_DIR / archivo_ordenado(), "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(registros)


def parse_valor(campo, texto):
    texto = (texto or "").strip()
    if not texto:
        raise ValueError("Debe introducir un dato a buscar.")
    if campo in CAMPOS_NUMERICOS:
        try:
            return int(texto.replace(",", ""))
        except ValueError:
            raise ValueError(
                f"El campo {campo} es numérico; '{texto}' no es un entero válido."
            )
    return texto


# ----------------------------------------------------------------------------
# GESTIÓN DE ESTADO Y SESIÓN
# ----------------------------------------------------------------------------
def iniciar_estado():
    por_defecto = dict(
        usuario_activo=None,
        hora_login=None,
        datos_originales=None,
        datos_actuales=None,
        dataset_activo=None,
        archivo_ordenado=None,
        error_carga=None,
        tiempo_carga_ns=None,
        campo_ordenado=None,
        conteo_metodos={nombre: 0 for nombre, _ in ALGORITMOS.values()},
        reporte_orden=None,
        ultima_orden=None,
        csv_guardado=False,
        aviso_csv=None,
        metodo_busqueda=None,
        resultado_busqueda=None,
        valores_campo={},
        historial_personal=[],
    )
    for k, v in por_defecto.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_entradas():
    for k in [
        k
        for k in st.session_state.keys()
        if k.startswith("valor_") or k.startswith("campo_sel_")
    ]:
        del st.session_state[k]


def cerrar_sesion():
    """Cierra la sesión y borra todo lo de este navegador (datos en RAM, Método Alfa personal, historial)."""
    st.session_state.clear()


def reiniciar_alfa_personal():
    st.session_state.conteo_metodos = {
        nombre: 0 for nombre, _ in ALGORITMOS.values()
    }


def guardar_datos_cargados(datos, t_ns, nombre_archivo):
    st.session_state.datos_originales = datos
    st.session_state.datos_actuales = list(datos)
    st.session_state.dataset_activo = nombre_archivo
    st.session_state.archivo_ordenado = f"ganado_{len(datos)}_ordenado.csv"
    st.session_state.csv_guardado = False  # el CSV ordenado es por dataset
    st.session_state.tiempo_carga_ns = t_ns
    st.session_state.campo_ordenado = None
    st.session_state.reporte_orden = None
    st.session_state.ultima_orden = None
    st.session_state.aviso_csv = None
    st.session_state.metodo_busqueda = None
    st.session_state.resultado_busqueda = None
    st.session_state.valores_campo = {}
    limpiar_entradas()
    registrar_evento(
        "Carga a RAM",
        f"Archivo: {nombre_archivo} | Cargados {fmt(len(datos))} registros en Lista de Diccionarios",
        t_ns,
    )


AVISO_SIN_DATOS = (
    "⚠️ Primero debe cargar los datos en memoria RAM: vaya a la Ventana 1 y presione "
    "uno de los botones 📥 Cargar Base de 15,000 / 30,000 Registros."
)


# ----------------------------------------------------------------------------
# ESTÉTICA GAMER / TELEMETRY HUD (CSS)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --bg-dark: #07090e;
  --bg-panel: rgba(16, 22, 34, 0.85);
  --cyan: #00f3ff;
  --neon-green: #00ff9d;
  --magenta: #ff0055;
  --purple: #b537f2;
  --gold: #ffcc00;
}

.stApp {
  background: radial-gradient(circle at 50% 0%, #131b2e 0%, #07090e 75%);
  color: #e2e8f0;
  font-family: 'Rajdhani', sans-serif;
}

.block-container { max-width: 1280px; padding-top: 2.5rem; }

h1, h2, h3, .hero-title, .cred-tit, .login-title {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 1px;
}

.hero {
  background: linear-gradient(135deg, rgba(10,15,28,0.95) 0%, rgba(18,28,48,0.9) 60%, rgba(0,243,255,0.12) 100%);
  border: 1px solid rgba(0, 243, 255, 0.45);
  border-left: 6px solid var(--cyan);
  border-radius: 16px;
  padding: 26px 32px;
  margin-bottom: 18px;
  box-shadow: 0 0 28px rgba(0, 243, 255, 0.15), inset 0 0 20px rgba(0, 243, 255, 0.05);
  position: relative;
  overflow: hidden;
}
.hero-sup {
  letter-spacing: 3px;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--cyan);
  text-transform: uppercase;
}
.hero-title {
  font-size: 1.85rem;
  font-weight: 900;
  line-height: 1.2;
  margin: 6px 0 8px;
  color: #ffffff;
  text-shadow: 0 0 14px rgba(0, 243, 255, 0.45);
}
.hero-sub { font-size: 1.05rem; color: #94a3b8; }
.hero-chips span {
  display: inline-block;
  background: rgba(0, 243, 255, 0.1);
  border: 1px solid rgba(0, 243, 255, 0.45);
  color: var(--cyan);
  border-radius: 6px;
  padding: 4px 12px;
  margin: 12px 8px 0 0;
  font-size: 0.82rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}

.user-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0, 255, 157, 0.07);
  border: 1px solid rgba(0, 255, 157, 0.35);
  border-radius: 12px;
  padding: 10px 18px;
  margin-bottom: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88rem;
}

.login-box {
  background: rgba(14, 20, 32, 0.92);
  border: 1px solid var(--cyan);
  border-top: 5px solid var(--cyan);
  border-radius: 18px;
  padding: 28px 32px;
  box-shadow: 0 0 35px rgba(0, 243, 255, 0.2);
  margin: 20px auto;
}
.login-title {
  color: var(--cyan);
  font-size: 1.45rem;
  font-weight: 800;
  margin-bottom: 6px;
}

/* Botones estilo HUD */
div.stButton > button, div.stDownloadButton > button {
  background: linear-gradient(135deg, #0b1528 0%, #102a45 100%);
  color: var(--cyan);
  border: 1px solid var(--cyan);
  border-radius: 8px;
  font-family: 'Orbitron', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  padding: 0.55rem 1.2rem;
  box-shadow: 0 0 12px rgba(0, 243, 255, 0.2);
  transition: all 0.2s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background: var(--cyan);
  color: #050811;
  box-shadow: 0 0 20px rgba(0, 243, 255, 0.7);
  transform: translateY(-2px);
}

/* Pestañas (Ventanas) */
button[data-baseweb="tab"] {
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.6px;
  padding: 10px 18px !important;
}

/* Tarjetas HUD */
.card {
  border-radius: 12px;
  padding: 16px 18px;
  margin: 6px 0 12px;
  background: var(--bg-panel);
  border: 1px solid rgba(0, 243, 255, 0.28);
  border-left: 5px solid var(--cyan);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
}
.card-verde {
  border-color: rgba(0, 255, 157, 0.3);
  border-left-color: var(--neon-green);
}
.card-dorado {
  border-color: rgba(255, 204, 0, 0.45);
  border-left-color: var(--gold);
  background: rgba(255, 204, 0, 0.06);
}
.card-marron {
  border-color: rgba(255, 0, 85, 0.4);
  border-left-color: var(--magenta);
  background: rgba(255, 0, 85, 0.06);
}
.card-ico { font-size: 1.45rem; line-height: 1; }
.card-tit {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: #94a3b8;
  margin-top: 6px;
}
.card-val {
  font-size: 1.6rem;
  font-weight: 800;
  line-height: 1.25;
  color: #ffffff;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
}
.card-pie { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

.barra {
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.1);
  margin-top: 10px;
  overflow: hidden;
}
.barra-in {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--cyan), var(--neon-green));
  box-shadow: 0 0 10px var(--cyan);
}

/* Credencial / Ficha Técnica HUD */
.cred {
  border-radius: 14px;
  margin: 14px 0;
  overflow: hidden;
  border: 1px solid var(--cyan);
  box-shadow: 0 0 25px rgba(0, 243, 255, 0.2);
  background: rgba(11, 17, 29, 0.92);
}
.cred-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  background: linear-gradient(90deg, #0b192c 0%, #112d4e 60%, #005f73 100%);
  border-bottom: 1px solid var(--cyan);
  color: #fff;
  padding: 14px 22px;
}
.cred-sup { font-size: 0.72rem; letter-spacing: 2px; font-weight: 700; color: var(--cyan); }
.cred-tit { font-size: 1.15rem; font-weight: 800; }
.cred-fila { font-size: 0.82rem; text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--neon-green); }
.cred-body { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 24px; padding: 18px 22px; }
.cred-item {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(0, 243, 255, 0.2);
}
.cred-ico { font-size: 1.55rem; width: 2.2rem; text-align: center; }
.cred-lbl { font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #94a3b8; }
.cred-val { font-size: 1.05rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #f8fafc; word-break: break-word; }
.cred-pie { padding: 10px 22px; font-size: 0.78rem; color: #94a3b8; border-top: 1px dashed rgba(0, 243, 255, 0.35); }
@media (max-width:700px){ .cred-body { grid-template-columns: 1fr; } .hero-title { font-size: 1.35rem; } }
</style>
"""


# ----------------------------------------------------------------------------
# COMPONENTES VISUALES
# ----------------------------------------------------------------------------
def mostrar_df(df, config=None):
    try:
        st.dataframe(
            df, hide_index=True, width="stretch", column_config=config
        )
    except TypeError:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config=config,
        )


def construir_tabla(registros, reg_inicial):
    filas = []
    for i, r in enumerate(registros):
        fila = {"Reg#": reg_inicial + i}
        for c in COLUMNAS:
            fila[ETIQUETA[c]] = r[c]
        filas.append(fila)
    return pd.DataFrame(filas)


def config_columnas():
    cfg = {
        ETIQUETA[c]: st.column_config.NumberColumn(format="%d")
        for c in CAMPOS_NUMERICOS
    }
    cfg["Reg#"] = st.column_config.NumberColumn(format="%d")
    return cfg


def tarjeta(icono, titulo, valor, pie="", color="verde", extra=""):
    return (
        f"<div class='card card-{color}'><div class='card-ico'>{icono}</div>"
        f"<div class='card-tit'>{html.escape(str(titulo))}</div>"
        f"<div class='card-val'>{html.escape(str(valor))}</div>"
        f"<div class='card-pie'>{html.escape(str(pie))}</div>{extra}</div>"
    )


def fila_tarjetas(tarjetas):
    cols = st.columns(len(tarjetas))
    for col, t in zip(cols, tarjetas):
        col.markdown(t, unsafe_allow_html=True)


def mostrar_ficha(registro, fila):
    items = ""
    for c in COLUMNAS:
        marca = "*" if c in CAMPOS_CLAVE.values() else ""
        items += (
            f"<div class='cred-item'><div class='cred-ico'>{ICONOS[c]}</div><div>"
            f"<div class='cred-lbl'>{c}{marca}</div>"
            f"<div class='cred-val'>{html.escape(str(registro[c]))}</div></div></div>"
        )
    st.markdown(
        "<div class='cred'><div class='cred-head'><div>"
        "<div class='cred-sup'>TELEMETRÍA ZOOSANITARIA · SINIIGA NACIONAL</div>"
        "<div class='cred-tit'>INFORMACIÓN DEL REGISTRO LOCALIZADO</div></div>"
        f"<div class='cred-fila'>POSICIÓN EN RAM<br><b>FILA #{fila}</b></div></div>"
        f"<div class='cred-body'>{items}</div>"
        "<div class='cred-pie'>⚡ Padrón Nacional de los 32 Estados de la República Mexicana · "
        "SENASICA / SADER / SIAP · Asistencia IA: Claude AI</div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# USUARIOS (usuarios.json con contraseñas cifradas con SHA-256 + sal)
# ----------------------------------------------------------------------------
USUARIO_INVITADO = "Invitado"


def normalizar_usuario(usuario):
    return " ".join((usuario or "").strip().split())


def hash_password(password, sal):
    return hashlib.sha256((sal + password).encode("utf-8")).hexdigest()


def leer_usuarios():
    """Devuelve el dict de usuarios ({} si aún no existe) o None si usuarios.json está dañado."""
    ruta = BASE_DIR / ARCHIVO_USUARIOS
    if not ruta.exists():
        return {}
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return datos if isinstance(datos, dict) else None
    except Exception:
        return None


def guardar_usuarios(usuarios):
    ruta = BASE_DIR / ARCHIVO_USUARIOS
    tmp = ruta.with_name(ARCHIVO_USUARIOS + ".tmp")
    tmp.write_text(json.dumps(usuarios, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ruta)  # escritura atómica


def registrar_usuario(usuario, password, confirmar):
    """Crea la cuenta. Devuelve (ok, mensaje)."""
    nombre = normalizar_usuario(usuario)
    if not nombre or not password or not confirmar:
        return False, "Complete los tres campos: Usuario, Contraseña y Confirmar Contraseña."
    if password != confirmar:
        return False, "Las contraseñas no coinciden."
    clave = nombre.casefold()
    if clave == USUARIO_INVITADO.casefold():
        return False, f"El nombre «{USUARIO_INVITADO}» está reservado para el modo invitado."
    usuarios = leer_usuarios()
    if usuarios is None:
        return False, f"No se puede leer '{ARCHIVO_USUARIOS}' (archivo dañado). Avise al administrador."
    if clave in usuarios:
        return False, f"El usuario «{nombre}» ya existe. Elija otro nombre."
    sal = secrets.token_hex(16)
    usuarios[clave] = {
        "usuario": nombre,
        "sal": sal,
        "sha256": hash_password(password, sal),
        "creado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    guardar_usuarios(usuarios)
    return True, nombre


def validar_credenciales(usuario, password):
    """Devuelve el nombre de usuario registrado si las credenciales coinciden, o None."""
    usuarios = leer_usuarios() or {}
    reg = usuarios.get(normalizar_usuario(usuario).casefold())
    if not reg or not password:
        return None
    esperado = reg.get("sha256", "")
    if hmac.compare_digest(hash_password(password, reg.get("sal", "")), esperado):
        return reg.get("usuario", normalizar_usuario(usuario))
    return None


# ----------------------------------------------------------------------------
# PANTALLA DE ACCESO: INICIAR SESIÓN / REGISTRAR USUARIO / INVITADO
# ----------------------------------------------------------------------------
def iniciar_sesion(nombre, detalle):
    """Abre una sesión nueva: Método Alfa personal en cero e historial vacío."""
    st.session_state.usuario_activo = nombre
    st.session_state.hora_login = datetime.now().strftime("%H:%M:%S")
    st.session_state.conteo_metodos = {n: 0 for n, _ in ALGORITMOS.values()}
    st.session_state.historial_personal = []
    registrar_evento("Inicio de Sesión", detalle, 0)


def entrar_como_invitado():
    iniciar_sesion(USUARIO_INVITADO, "Ingreso como Invitado (sesión temporal)")


def pantalla_login():
    st.markdown(CSS, unsafe_allow_html=True)
    _, col_centro, _ = st.columns([1, 1.6, 1])
    with col_centro:
        st.markdown(
            "<div class='login-box'>"
            "<div class='hero-sup'>SISTEMA NACIONAL SINIIGA · SENASICA</div>"
            "<div class='login-title'>🔐 CONTROL DE ACCESO OPERATIVO</div>"
            "<p style='color:#94a3b8;margin-bottom:0;'>Inicie sesión con su cuenta, regístrese como nuevo "
            "operador o entre como invitado para habilitar su consola de telemetría en memoria RAM "
            "y contabilizar su Método Alfa personal.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        tab_login, tab_registro = st.tabs(["🔑 Iniciar Sesión", "📝 Registrar Usuario"])

        with tab_login:
            with st.form("form_login", clear_on_submit=False):
                usuario = st.text_input("Usuario", key="login_usuario")
                password = st.text_input("Contraseña", type="password", key="login_password")
                entrar = st.form_submit_button(
                    "🚀 INICIAR SESIÓN", key="btn_login", use_container_width=True
                )
            if entrar:
                if not normalizar_usuario(usuario) or not password:
                    st.error("⚠️ Ingrese su Usuario y su Contraseña.")
                elif leer_usuarios() is None:
                    st.error(f"No se puede leer '{ARCHIVO_USUARIOS}' (archivo dañado). Avise al administrador.")
                else:
                    nombre = validar_credenciales(usuario, password)
                    if nombre is None:
                        st.error("⚠️ Usuario o contraseña incorrectos.")
                    else:
                        iniciar_sesion(nombre, "Ingreso autorizado a la consola")
                        st.rerun()

        with tab_registro:
            with st.form("form_registro", clear_on_submit=True):
                nuevo_usuario = st.text_input("Usuario", key="reg_usuario")
                nueva_pass = st.text_input("Contraseña", type="password", key="reg_password")
                confirmar = st.text_input("Confirmar Contraseña", type="password", key="reg_confirmar")
                crear = st.form_submit_button(
                    "✅ CREAR CUENTA", key="btn_registrar", use_container_width=True
                )
            if crear:
                ok, mensaje = registrar_usuario(nuevo_usuario, nueva_pass, confirmar)
                if ok:
                    st.success(
                        f"✅ Cuenta «{mensaje}» creada correctamente. Ahora vaya a la pestaña "
                        "«Iniciar Sesión» para entrar."
                    )
                else:
                    st.error(f"⚠️ {mensaje}")

        st.button(
            "👤 Iniciar Sesión como Invitado",
            key="btn_invitado",
            on_click=entrar_como_invitado,
            use_container_width=True,
        )
        st.caption("El modo invitado usa una sesión temporal con su Método Alfa personal en cero.")


# ----------------------------------------------------------------------------
# ENCABEZADO Y BARRA DE USUARIO
# ----------------------------------------------------------------------------
def seccion_encabezado():
    st.markdown(CSS, unsafe_allow_html=True)

    c_info, c_btn = st.columns([4, 1])
    with c_info:
        st.markdown(
            f"<div class='user-bar'>"
            f"<span>🟢 <b>OPERADOR ACTIVO:</b> {html.escape(st.session_state.usuario_activo)}</span>"
            f"<span>🕒 Ingreso: {st.session_state.hora_login}</span>"
            f"<span>🤖 IA: Claude AI (Anthropic)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c_btn:
        st.button(
            "🚪 Cerrar Sesión",
            key="btn_logout",
            on_click=cerrar_sesion,
            use_container_width=True,
        )

    datos_hero = st.session_state.datos_actuales
    if datos_hero:
        m_hero = tamano_muestra(len(datos_hero))
        chip_bovinos = f"🐄 {fmt(len(datos_hero))} BOVINOS"
        chip_muestra = f"📊 MUESTRA {m_hero} + {m_hero}"
    else:
        chip_bovinos = "🐄 BASE: 15,000 O 30,000 BOVINOS"
        chip_muestra = f"📊 MUESTRA {MUESTRA_REGISTROS} + {MUESTRA_REGISTROS}"
    st.markdown(
        "<div class='hero'><div class='hero-sup'>SADER · SENASICA · SINIIGA · PADRÓN NACIONAL (32 ESTADOS)</div>"
        "<div class='hero-title'>PROYECTO 2: RASTREO E INVENTARIO BOVINO PARA EXPORTACIÓN</div>"
        "<div class='hero-sub'>Consola de Telemetría en Memoria RAM · Algoritmos de Ordenación y Búsqueda</div>"
        f"<div class='hero-chips'><span>{chip_bovinos}</span><span>🗺️ 32 ESTADOS DE LA REPÚBLICA</span>"
        f"<span>⏱️ NANOSEGUNDOS (ns)</span><span>{chip_muestra}</span></div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# VENTANA 1: CARGA EN RAM Y MUESTRA (60 + 60)
# ----------------------------------------------------------------------------
def cargar_dataset(nombre_archivo):
    """Callback de los botones de carga: se ejecuta antes de dibujar la página."""
    st.session_state.error_carga = None
    ruta = BASE_DIR / nombre_archivo
    if not ruta.exists():
        st.session_state.error_carga = f"No se encontró '{nombre_archivo}' junto a app.py."
        return
    try:
        datos, t_ns = cargar_datos_ram(str(ruta))
    except Exception as e:
        st.session_state.error_carga = f"No se pudo cargar '{nombre_archivo}': {e}"
        return
    guardar_datos_cargados(datos, t_ns, nombre_archivo)


def seccion_carga_y_muestra():
    datos = st.session_state.datos_actuales
    if datos:
        m = tamano_muestra(len(datos))
        st.subheader(f"🖥️ Infraestructura de Carga en Memoria RAM y Muestra ({m} + {m})")
    else:
        st.subheader("🖥️ Infraestructura de Carga en Memoria RAM y Muestra")

    ruta_contexto = BASE_DIR / ARCHIVO_CONTEXTO
    with st.expander(
        "📄 Ver Contexto Institucional del Proyecto (SENASICA / SADER)",
        expanded=False,
    ):
        if ruta_contexto.exists():
            texto_ctx = ruta_contexto.read_text(encoding="utf-8-sig")
            for linea in texto_ctx.splitlines():
                linea = linea.strip()
                if not linea:
                    continue
                st.write(linea)
        else:
            st.info(
                "Padrón Nacional Zoosanitario de los 32 Estados de la República Mexicana para exportación bovina."
            )

    # Dos botones independientes: si falta un CSV, solo ese botón queda deshabilitado con su advertencia.
    columnas = st.columns(2)
    for col, (nombre, etiqueta) in zip(columnas, DATASETS):
        with col:
            existe = (BASE_DIR / nombre).exists()
            st.button(
                f"📥 Cargar Base de {etiqueta} Registros ({nombre})",
                key=f"btn_cargar_{nombre}",
                on_click=cargar_dataset,
                args=(nombre,),
                disabled=not existe,
                use_container_width=True,
            )
            if st.session_state.dataset_activo == nombre:
                st.caption("✅ Base activa en memoria RAM")
            if not existe:
                st.warning(
                    f"No se encontró '{nombre}' junto a app.py. Colóquelo en la carpeta de la app "
                    "o súbalo aquí para habilitar este botón."
                )
                subido = st.file_uploader(
                    f"Subir {nombre}", type="csv", key=f"subir_{nombre}"
                )
                if subido is not None:
                    (BASE_DIR / nombre).write_bytes(subido.getvalue())
                    st.rerun()

    if st.session_state.error_carga:
        st.error(st.session_state.error_carga)

    if datos is None:
        st.info(
            "Los datos todavía no están en memoria RAM. Presione uno de los dos botones superiores "
            "para leer el archivo y cargarlo a la estructura de Lista de Diccionarios."
        )
        return

    n = len(datos)
    t = st.session_state.tiempo_carga_ns
    archivo = st.session_state.dataset_activo
    estados_unicos = len({r["estado_origen"] for r in datos})

    fila_tarjetas([
        tarjeta(
            "🐄",
            "Registros en RAM",
            fmt(n),
            f"Archivo: {archivo}",
            "verde",
        ),
        tarjeta(
            "🗺️",
            "Cobertura Nacional",
            f"{estados_unicos} Estados",
            "República Mexicana completa",
            "verde",
        ),
        tarjeta(
            "🗂️",
            "Columnas",
            len(datos[0]),
            "3 Claves (*) + 7 Informativas",
            "verde",
        ),
        tarjeta(
            "⏱️",
            "Tiempo de Carga RAM",
            f"{fmt(t)} ns",
            "Reloj CPU perf_counter_ns",
            "dorado",
        ),
    ])
    st.success(
        f"Se realizó la carga de los {fmt(n)} registros con sus {len(datos[0])} columnas a la estructura "
        f"[Lista de Diccionarios en RAM] y le tomó [{t}] nanosegundos completar la acción."
    )

    campo = st.session_state.campo_ordenado
    st.caption(
        f"⚡ Orden actual en RAM: ordenado por [{campo}]"
        if campo
        else "⚡ Orden actual en RAM: original del archivo CSV (sin ordenar). Los campos clave llevan *."
    )
    m = tamano_muestra(n)
    cfg = config_columnas()
    st.markdown(f"#### 🔹 Primeros {m} registros en RAM (Reg# 1 a {m})")
    mostrar_df(construir_tabla(datos[:m], 1), cfg)

    st.markdown(
        f"#### 🔸 Últimos {m} registros en RAM (Reg# {max(1, n - m + 1)} a {n})"
    )
    ultimos = datos[-m:]
    mostrar_df(construir_tabla(ultimos, n - len(ultimos) + 1), cfg)


# ----------------------------------------------------------------------------
# VENTANA 2: ORDENACIÓN Y MÉTODO ALFA (USUARIO VS GLOBAL)
# ----------------------------------------------------------------------------
def ejecutar_ordenacion(clave, automatica=False):
    datos = list(st.session_state.datos_originales)
    n = len(datos)
    campo = CAMPOS_CLAVE[clave]
    nombre, funcion = ALGORITMOS[clave]

    ordenada, t_ns, comps, movs = funcion(datos, campo)

    texto = (
        f"Se ordenaron {fmt(n)} registros por el campo [{campo}] usando el método [{nombre}] "
        f"en [{t_ns}] nanosegundos. Comparaciones: [{comps}], Intercambios/Movimientos: [{movs}]"
    )
    st.session_state.datos_actuales = ordenada
    st.session_state.campo_ordenado = campo
    st.session_state.conteo_metodos[nombre] += 1
    incrementar_alfa_global(nombre)

    st.session_state.reporte_orden = texto
    st.session_state.ultima_orden = dict(
        campo=campo, nombre=nombre, t_ns=t_ns, comps=comps, movs=movs
    )
    st.session_state.resultado_busqueda = None

    escribir_bitacora(
        BITACORA_ORDENACION,
        (
            "[Ordenación automática previa a una Búsqueda Binaria]\n"
            if automatica
            else ""
        )
        + texto,
    )
    registrar_evento(
        "Ordenación Automática" if automatica else "Disparador Ordenación",
        f"Método: {nombre} | Campo: {campo} | Comps: {fmt(comps)} | Movs: {fmt(movs)}",
        t_ns,
    )

    if not st.session_state.csv_guardado:
        guardar_csv_ordenado(ordenada)
        st.session_state.csv_guardado = True
        st.session_state.aviso_csv = (
            f"Primera ejecución: se guardó '{archivo_ordenado()}' con los {fmt(n)} registros "
            f"ordenados por [{campo}]."
        )
    return nombre, t_ns


def disparar_ordenacion():
    if st.session_state.datos_originales is None:
        return
    ejecutar_ordenacion(random.choice([1, 2, 3]))


def renderizar_bloque_alfa(conteo_dict, titulo_bloque, sub_etiqueta):
    st.markdown(f"#### {titulo_bloque}")
    total = sum(conteo_dict.values())
    maximo = max(conteo_dict.values()) if conteo_dict else 0
    alfas = (
        [nombre for nombre, v in conteo_dict.items() if v == maximo]
        if maximo > 0
        else []
    )
    cols = st.columns(len(ALGORITMOS))
    for col, (clave, (nombre, _)) in zip(cols, ALGORITMOS.items()):
        veces = conteo_dict.get(nombre, 0)
        es_alfa = nombre in alfas
        ancho = round(veces / maximo * 100) if maximo else 0
        parte = round(veces / total * 100) if total else 0
        col.markdown(
            tarjeta(
                "👑" if es_alfa else "⚙️",
                nombre + (" · ALFA" if es_alfa else ""),
                veces,
                f"{CAMPOS_CLAVE[clave]} · {parte}% ({sub_etiqueta})",
                "dorado" if es_alfa else "verde",
                extra=f"<div class='barra'><div class='barra-in' style='width:{ancho}%'></div></div>",
            ),
            unsafe_allow_html=True,
        )
    if not alfas:
        st.caption(f"Sin ejecuciones registradas aún en {sub_etiqueta}.")
    elif len(alfas) == 1:
        st.success(
            f"👑 Método Alfa ({sub_etiqueta}): **{alfas[0]}** con **{maximo}** ejecución(es) de {total} totales."
        )
    else:
        st.info(
            f"👑 Empate de Método Alfa ({sub_etiqueta}) entre **{', '.join(alfas)}** ({maximo} c/u)."
        )


def seccion_ordenacion_y_alfa():
    st.subheader("⚡ Disparador Aleatorio de Ordenación en RAM")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.button(
            "🎲 Disparador Aleatorio de Ordenación",
            key="btn_ordenar",
            on_click=disparar_ordenacion,
            use_container_width=True,
        )
    with c2:
        st.button(
            "🔄 Reiniciar Mi Método Alfa",
            key="btn_reset_alfa",
            on_click=reiniciar_alfa_personal,
            use_container_width=True,
        )

    if st.session_state.datos_originales is None:
        st.warning(AVISO_SIN_DATOS)
        return

    u = st.session_state.ultima_orden
    if u:
        fila_tarjetas([
            tarjeta(
                "🎯", "Campo Clave · Método", u["campo"], u["nombre"], "verde"
            ),
            tarjeta(
                "⏱️", "Tiempo CPU (ns)", fmt(u["t_ns"]), "nanosegundos", "dorado"
            ),
            tarjeta(
                "🔍", "Comparaciones", fmt(u["comps"]), "entre claves", "verde"
            ),
            tarjeta(
                "🔁",
                "Movimientos",
                fmt(u["movs"]),
                "asignaciones en RAM",
                "marron",
            ),
        ])
        st.success(st.session_state.reporte_orden)
        if st.session_state.aviso_csv:
            st.info(st.session_state.aviso_csv)
    else:
        st.caption(
            "Presione el disparador para sortear al azar el campo clave (siniiga_id, peso_kg o fecha_vacunacion) y ejecutar su algoritmo."
        )

    st.divider()
    st.subheader("🏆 Telemetría del Método Alfa (Por Usuario vs. Global)")

    renderizar_bloque_alfa(
        st.session_state.conteo_metodos,
        f"👤 Registro Individual de: {st.session_state.usuario_activo}",
        "Sesión Actual",
    )
    st.write("")
    conteo_global = leer_alfa_global()
    renderizar_bloque_alfa(
        conteo_global,
        "🌐 Registro Global Acumulado (Todos los Usuarios de la Página)",
        "Histórico Global",
    )


# ----------------------------------------------------------------------------
# VENTANA 3: DISPARADOR ALEATORIO DE BÚSQUEDA
# ----------------------------------------------------------------------------
def disparar_busqueda():
    if st.session_state.datos_originales is None:
        return
    st.session_state.metodo_busqueda = random.choice(
        ["Búsqueda Binaria", "Búsqueda Lineal"]
    )
    st.session_state.resultado_busqueda = None
    limpiar_entradas()


def ejecutar_busqueda(
    metodo, campo, valor, valor_txt, estado_orden=None, aviso=None
):
    datos = st.session_state.datos_actuales
    n = len(datos)
    if metodo == "Búsqueda Binaria":
        fila, reg, todos, t_ns, comps = busqueda_binaria(datos, campo, valor)
        cabecera = (
            "[FASE 3: DISPARADOR ALEATORIO DE BÚSQUEDA]\n"
            f"Operador en turno: {st.session_state.usuario_activo}\n"
            "Activando generador de aleatoriedad para la búsqueda del sistema...\n"
            f"-> Método de búsqueda seleccionado aleatoriamente: [{metodo}]\n"
            "[VALIDACIÓN DE SEGURIDAD]: Verificando estado físico de la memoria...\n"
            f"El campo clave requerido para la búsqueda es: [{campo}]\n"
            f"Estado del campo en memoria: {estado_orden or 'YA ORDENADO PREVIAMENTE.'}\n"
            f"Introduzca el dato a buscar ({campo}): {valor_txt}\n"
            "Ejecutando algoritmo de Búsqueda Binaria...\n"
        )
    else:
        fila, reg, todos, t_ns, comps = busqueda_lineal(datos, campo, valor)
        cabecera = (
            f"Operador en turno: {st.session_state.usuario_activo}\n"
            "Activando generador de aleatoriedad para la búsqueda del sistema...\n"
            f"-> Método de búsqueda seleccionado aleatoriamente: [{metodo}]\n"
            f"El campo seleccionado para la búsqueda es: [{campo}]\n"
            f"Introduzca el dato a buscar del campo seleccionado ({campo}): {valor_txt}\n"
            "Ejecutando algoritmo de Búsqueda Lineal...\n"
        )
    if fila is not None:
        resultado = f"Dato encontrado en la posición indexada de memoria RAM [Fila {fila}].\n"
    else:
        resultado = "Dato NO encontrado: ningún registro de la memoria RAM coincide con el valor buscado.\n"
    benchmark = (
        f">>> BENCHMARKING DE BÚSQUEDA (Guardado en {BITACORA_BUSQUEDA}) <<<\n"
        f'"Se realizó la búsqueda en {fmt(n)} registros por el campo [{campo}] usando el método de '
        f'búsqueda [{metodo}] en [{t_ns}] nanosegundos. Comparaciones: [{comps}]"'
    )
    texto = cabecera + resultado + benchmark
    escribir_bitacora(BITACORA_BUSQUEDA, texto)
    registrar_evento(
        f"Búsqueda ({metodo})",
        f"Campo: {campo} | Buscado: '{valor_txt}' | {'Fila ' + str(fila) if fila else 'No encontrado'} | Comps: {fmt(comps)}",
        t_ns,
    )
    st.session_state.resultado_busqueda = dict(
        texto=texto,
        fila=fila,
        registro=reg,
        todos=todos,
        campo=campo,
        metodo=metodo,
        t_ns=t_ns,
        comps=comps,
        aviso=aviso,
    )


MAX_VALORES_AYUDA = 35


def valores_disponibles(campo):
    cache = st.session_state.valores_campo
    if campo not in cache:
        vals = {
            str(r[campo]).strip() for r in st.session_state.datos_originales
        }
        clave = (
            (lambda v: int(v))
            if campo in CAMPOS_NUMERICOS
            else normalizar_texto
        )
        cache[campo] = sorted((v for v in vals if v), key=clave)
    return cache[campo]


def texto_valores_disponibles(campo):
    vals = valores_disponibles(campo)
    texto = (
        f"Valores disponibles en el padrón nacional ({len(vals)} distintos): "
        + ", ".join(vals[:MAX_VALORES_AYUDA])
    )
    if len(vals) > MAX_VALORES_AYUDA:
        texto += f" … (y {len(vals) - MAX_VALORES_AYUDA} más)"
    return texto


FORMATOS_CLAVE = {
    "siniiga_id": (
        "Formato requerido: Número entero de 8 dígitos sin espacios ni guiones "
        "(Rango: 10000000 a 99999999)."
    ),
    "peso_kg": (
        "Formato requerido: Número entero expresado en kilogramos sin decimales "
        "(Rango en padrón: 280 a 800)."
    ),
    "fecha_vacunacion": (
        "Formato requerido: Número entero en formato AAAAMMDD sin guiones ni diagonales "
        "(Ej. 20260415)."
    ),
}


def ejemplos_reales(campo, cantidad=5):
    datos = st.session_state.datos_actuales
    n = len(datos)
    indices = [0, 14, n // 4, n // 2, n - 1]
    paso = max(1, n // 11)
    indices += list(range(paso, n, paso))
    valores = []
    for i in indices:
        if 0 <= i < n and datos[i][campo] not in valores:
            valores.append(datos[i][campo])
        if len(valores) == cantidad:
            break
    return valores


def formato_campo(campo):
    if campo in FORMATOS_CLAVE:
        return FORMATOS_CLAVE[campo]
    if campo == "edad_meses":
        edades = [r["edad_meses"] for r in st.session_state.datos_originales]
        return (
            "Formato requerido: Número entero de meses sin decimales "
            f"(Rango en padrón: {min(edades)} a {max(edades)})."
        )
    return (
        "Formato requerido: Texto (Padrón de los 32 estados de la República). "
        "No distingue mayúsculas, acentos ni espacios extra y acepta coincidencias parciales."
    )


def guia_campo(campo):
    ejemplos = ejemplos_reales(campo)
    st.info(
        f"{formato_campo(campo)}\n\n"
        f"Ejemplos reales en memoria RAM listos para probar: [{', '.join(str(v) for v in ejemplos)}]"
    )
    return ejemplos


def limpiar_busqueda():
    limpiar_entradas()
    st.session_state.resultado_busqueda = None


def seccion_busqueda():
    st.subheader("🔍 Radar y Disparador Aleatorio de Búsqueda")
    st.button(
        "🎲 Disparador Aleatorio de Búsqueda",
        key="btn_buscar_azar",
        on_click=disparar_busqueda,
    )
    if st.session_state.datos_originales is None:
        st.warning(AVISO_SIN_DATOS)
        return

    metodo = st.session_state.metodo_busqueda
    if metodo is None:
        st.caption(
            "Sortea al azar el método (Búsqueda Lineal o Búsqueda Binaria) y habilita el selector de campo."
        )
        return
    st.markdown(f"### 🎯 Método sorteado: `{metodo}`")

    ordenado = st.session_state.campo_ordenado
    if metodo == "Búsqueda Binaria":
        opciones = list(CAMPOS_CLAVE.values())
        idx = opciones.index(ordenado) if ordenado in opciones else 0
        campo = st.selectbox(
            "Seleccione el Campo Clave donde desea buscar",
            opciones,
            index=idx,
            key="campo_sel_binaria",
        )
        if campo != ordenado:
            nombre = ALGORITMOS[CLAVE_DE_CAMPO[campo]][0]
            actual = (
                f"(ahora está ordenado por [{ordenado}])"
                if ordenado
                else "(aún no se ha ordenado nada)"
            )
            st.warning(
                f"El campo [{campo}] no está ordenado en RAM {actual}. Al presionar Buscar se ordenará "
                f"automáticamente con {nombre} y enseguida se ejecutará la Búsqueda Binaria."
            )
        else:
            st.caption(f"✔ [{campo}] ya está ordenado en memoria RAM.")
    else:
        opciones = list(COLUMNAS)
        idx = opciones.index(ordenado) if ordenado in opciones else 0
        campo = st.selectbox(
            "Seleccione el Campo (Clave o Informativo) donde desea buscar",
            opciones,
            index=idx,
            key="campo_sel_lineal",
        )

    ejemplos = guia_campo(campo)
    if campo not in CAMPOS_CLAVE.values():
        st.caption(texto_valores_disponibles(campo))

    if campo in CAMPOS_NUMERICOS:
        valor = st.number_input(
            f"Introduzca el dato numérico a buscar ({campo})",
            min_value=0,
            value=int(ejemplos[0]),
            step=1,
            format="%d",
            key=f"valor_{campo}",
        )
    else:
        valor_txt = st.text_input(
            f"Introduzca el texto a buscar ({campo})", key=f"valor_{campo}"
        )

    b1, b2, _ = st.columns([1.2, 2, 3])
    buscar = b1.button(
        "🔍 Buscar Ahora", key="btn_buscar", use_container_width=True
    )
    b2.button(
        "🧹 Limpiar / Reiniciar",
        key="btn_limpiar",
        on_click=limpiar_busqueda,
        use_container_width=True,
    )

    if buscar:
        if campo in CAMPOS_NUMERICOS:
            valor_buscado, texto_valor = int(valor), str(int(valor))
        else:
            try:
                valor_buscado, texto_valor = (
                    parse_valor(campo, valor_txt),
                    valor_txt.strip(),
                )
            except ValueError as e:
                st.error(str(e))
                return
        estado, aviso = None, None
        reordenado = (
            metodo == "Búsqueda Binaria"
            and campo != st.session_state.campo_ordenado
        )
        if reordenado:
            nombre, t_orden = ejecutar_ordenacion(
                CLAVE_DE_CAMPO[campo], automatica=True
            )
            estado = (
                f"NO ESTABA ORDENADO. Se ordenó automáticamente con [{nombre}] "
                f"en [{t_orden}] nanosegundos."
            )
            aviso = (
                f"Se ordenó automáticamente por [{campo}] con {nombre} en {fmt(t_orden)} ns "
                "antes de ejecutar la Búsqueda Binaria."
            )
        ejecutar_busqueda(
            metodo, campo, valor_buscado, texto_valor, estado, aviso
        )
        if reordenado:
            st.rerun()

    res = st.session_state.resultado_busqueda
    if res and res["campo"] == campo and res["metodo"] == metodo:
        if res["aviso"]:
            st.info("🔄 " + res["aviso"])
        fila_tarjetas([
            tarjeta(
                "🔎", "Método · Campo", res["metodo"], res["campo"], "verde"
            ),
            tarjeta(
                "⏱️",
                "Tiempo Búsqueda (ns)",
                fmt(res["t_ns"]),
                "nanosegundos",
                "dorado",
            ),
            tarjeta(
                "⚖️",
                "Comparaciones",
                fmt(res["comps"]),
                "evaluadas en RAM",
                "marron",
            ),
        ])
        st.code(res["texto"], language="text")
        if res["fila"] is None:
            st.warning(
                "No se localizó ningún registro con ese dato en la memoria RAM."
            )
        else:
            mostrar_ficha(res["registro"], res["fila"])
            if len(res["todos"]) > 1:
                st.info(
                    f"Se encontraron {len(res['todos'])} registros con ese valor; "
                    "la credencial muestra la primera ocurrencia localizada."
                )
                with st.expander(
                    f"Ver todas las coincidencias ({len(res['todos'])} registros)"
                ):
                    filas = [
                        {"Fila": f, **{c: r[c] for c in COLUMNAS}}
                        for f, r in res["todos"]
                    ]
                    mostrar_df(pd.DataFrame(filas))


# ----------------------------------------------------------------------------
# VENTANA 4: CENTRO DE BITÁCORAS (PERSONAL VS GLOBAL) Y DESCARGAS
# ----------------------------------------------------------------------------
def seccion_bitacoras_y_descargas():
    st.subheader("📜 Centro de Bitácoras y Registro de Actividad")

    sub_p, sub_g, sub_d = st.tabs([
        "👤 Mi Historial (Sesión Actual)",
        "🌐 Historial Global (Todos los Visitantes)",
        "⬇️ Descarga de Archivos Oficiales",
    ])

    with sub_p:
        st.markdown(
            f"#### Registro de acciones de **{st.session_state.usuario_activo}**"
        )
        hist = st.session_state.historial_personal
        if not hist:
            st.info("Aún no ha ejecutado acciones en esta sesión.")
        else:
            mostrar_df(pd.DataFrame(hist))
            if st.button("🗑️ Limpiar Mi Historial Personal", key="btn_clr_hist"):
                st.session_state.historial_personal = []
                st.rerun()

    with sub_g:
        st.markdown(
            "#### Bitácora Global de Todos los Usuarios que Han Entrado a la Página"
        )
        ruta_g = BASE_DIR / BITACORA_GLOBAL
        if ruta_g.exists():
            lineas = [
                l.strip()
                for l in ruta_g.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            st.metric(
                "Total de Acciones Globales Registradas", fmt(len(lineas))
            )
            st.code(
                "\n".join(reversed(lineas[-150:])),
                language="text",
            )
        else:
            st.info("Aún no hay registros globales en el servidor.")

    with sub_d:
        st.markdown("#### Descargar Bitácoras y CSV Ordenado")
        archivos = [
            (BITACORA_ORDENACION, "text/plain", "Bitácora de Ordenación"),
            (BITACORA_BUSQUEDA, "text/plain", "Bitácora de Búsqueda"),
            (BITACORA_GLOBAL, "text/plain", "Bitácora Global de Usuarios"),
            (archivo_ordenado(), "text/csv", "CSV Ordenado"),
        ]
        cols = st.columns(2)
        for i, (nombre, mime, etiqueta) in enumerate(archivos):
            ruta = BASE_DIR / nombre
            col = cols[i % 2]
            if ruta.exists():
                col.download_button(
                    f"⬇️ Descargar {etiqueta} ({nombre})",
                    data=ruta.read_bytes(),
                    file_name=nombre,
                    mime=mime,
                    key=f"dl_{nombre}",
                    use_container_width=True,
                )
            else:
                col.caption(
                    f"⏳ {etiqueta} (`{nombre}`): se habilitará en cuanto ejecute la acción correspondiente."
                )


# ----------------------------------------------------------------------------
# FLUJO PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Telemetría Bovina SENASICA",
        page_icon="🐄",
        layout="wide",
    )
    iniciar_estado()

    if not st.session_state.usuario_activo:
        pantalla_login()
        return

    seccion_encabezado()

    v1, v2, v3, v4 = st.tabs([
        "🖥️ 1. Carga RAM y Muestra",
        "⚡ 2. Ordenación y Método Alfa",
        "🔍 3. Radar de Búsqueda",
        "📜 4. Bitácoras y Descargas",
    ])

    with v1:
        seccion_carga_y_muestra()
    with v2:
        seccion_ordenacion_y_alfa()
    with v3:
        seccion_busqueda()
    with v4:
        seccion_bitacoras_y_descargas()


if __name__ == "__main__":
    main()
