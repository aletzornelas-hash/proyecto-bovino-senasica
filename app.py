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
        extra = [c for c in encabezados if c not in COLUMNAS]
        detalle = []
        if faltantes:
            detalle.append(f"faltan columnas: {faltantes}")
        if extra:
            detalle.append(f"sobran columnas: {extra}")
        raise ValueError(
            "Estructura de columnas incorrecta. Este sistema solo acepta el padrón bovino "
            f"oficial de 10 columnas ({'; '.join(detalle)})."
        )

    datos = []
    for idx, fila in enumerate(lector, start=2):
        reg = {}
        for col in COLUMNAS:
            val = fila.get(col)
            if val is None or str(val).strip() == "":
                raise ValueError(
                    f"Dato vacío o corrupto en la fila {idx}, columna '{col}'."
                )
            if col in CAMPOS_NUMERICOS:
                try:
                    reg[col] = int(str(val).strip())
                except ValueError:
                    raise ValueError(
                        f"Tipo de dato inválido en la fila {idx}, columna '{col}': "
                        f"se esperaba un número entero y se recibió '{val}'."
                    )
            else:
                reg[col] = str(val).strip()
        datos.append(reg)

    tiempo_ns = time.perf_counter_ns() - inicio

    if len(datos) not in TAMANOS_PERMITIDOS:
        raise ValueError(
            f"Cantidad de registros no permitida: el archivo contiene {fmt(len(datos))} registros. "
            "Solo se aceptan archivos oficiales de exactamente 15,000 o 30,000 registros."
        )

    return datos, tiempo_ns


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
        cantidad_muestra=60,
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
    st.session_state.csv_guardado = False
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
    "⚠️ Primero debe cargar los datos en memoria RAM: seleccione «🖥️ 1. Carga RAM y Muestra» "
    "en el panel izquierdo y suba o cargue un archivo válido de 15,000 o 30,000 registros."
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

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #090d16 0%, #0d1526 100%);
  border-right: 1px solid rgba(0, 243, 255, 0.35);
}

.sidebar-badge {
  background: rgba(0, 243, 255, 0.08);
  border: 1px solid rgba(0, 243, 255, 0.4);
  border-left: 4px solid var(--cyan);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
}

.block-container { max-width: 1280px; padding-top: 2rem; }

h1, h2, h3, .hero-title, .cred-tit, .login-title {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 1px;
}

.hero {
  background: linear-gradient(135deg, rgba(10,15,28,0.95) 0%, rgba(18,28,48,0.9) 60%, rgba(0,243,255,0.12) 100%);
  border: 1px solid rgba(0, 243, 255, 0.45);
  border-left: 6px solid var(--cyan);
  border-radius: 16px;
  padding: 22px 28px;
  margin-bottom: 18px;
  box-shadow: 0 0 28px rgba(0, 243, 255, 0.15), inset 0 0 20px rgba(0, 243, 255, 0.05);
}
.hero-sup {
  letter-spacing: 3px;
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--cyan);
  text-transform: uppercase;
}
.hero-title {
  font-size: 1.65rem;
  font-weight: 900;
  line-height: 1.2;
  margin: 6px 0 8px;
  color: #ffffff;
  text-shadow: 0 0 14px rgba(0, 243, 255, 0.45);
}
.hero-sub { font-size: 1rem; color: #94a3b8; }
.hero-chips span {
  display: inline-block;
  background: rgba(0, 243, 255, 0.1);
  border: 1px solid rgba(0, 243, 255, 0.45);
  color: var(--cyan);
  border-radius: 6px;
  padding: 4px 12px;
  margin: 10px 8px 0 0;
  font-size: 0.8rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
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
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.7px;
  padding: 0.52rem 1.1rem;
  box-shadow: 0 0 12px rgba(0, 243, 255, 0.2);
  transition: all 0.2s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background: var(--cyan);
  color: #050811;
  box-shadow: 0 0 20px rgba(0, 243, 255, 0.7);
  transform: translateY(-2px);
}

button[data-baseweb="tab"] {
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.84rem !important;
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
.card-morado {
  border-color: rgba(181, 55, 242, 0.45);
  border-left-color: var(--purple);
  background: rgba(181, 55, 242, 0.07);
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
  font-size: 1.45rem;
  font-weight: 800;
  line-height: 1.25;
  color: #ffffff;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-word;
}
.card-pie { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }

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
        "SENASICA / SADER / SIAP · Asistencia IA: Gemini &amp; Claude AI</div></div>",
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
    tmp.write_text(
        json.dumps(usuarios, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, ruta)


def registrar_usuario(usuario, password, confirmar):
    nombre = normalizar_usuario(usuario)
    if not nombre or not password or not confirmar:
        return (
            False,
            "Complete los tres campos: Usuario, Contraseña y Confirmar Contraseña.",
        )
    if password != confirmar:
        return False, "Las contraseñas no coinciden."
    clave = nombre.casefold()
    if clave == USUARIO_INVITADO.casefold():
        return (
            False,
            f"El nombre «{USUARIO_INVITADO}» está reservado para el modo invitado.",
        )
    usuarios = leer_usuarios()
    if usuarios is None:
        return (
            False,
            f"No se puede leer '{ARCHIVO_USUARIOS}' (archivo dañado). Avise al administrador.",
        )
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
    usuarios = leer_usuarios() or {}
    reg = usuarios.get(normalizar_usuario(usuario).casefold())
    if not reg or not password:
        return None
    esperado = reg.get("sha256", "")
    if hmac.compare_digest(
        hash_password(password, reg.get("sal", "")), esperado
    ):
        return reg.get("usuario", normalizar_usuario(usuario))
    return None


# ----------------------------------------------------------------------------
# PANTALLA DE ACCESO: INICIAR SESIÓN / REGISTRAR USUARIO / INVITADO
# ----------------------------------------------------------------------------
def iniciar_sesion(nombre, detalle):
    st.session_state.usuario_activo = nombre
    st.session_state.hora_login = datetime.now().strftime("%H:%M:%S")
    st.session_state.conteo_metodos = {n: 0 for n, _ in ALGORITMOS.values()}
    st.session_state.historial_personal = []
    registrar_evento("Inicio de Sesión", detalle, 0)


def entrar_como_invitado():
    iniciar_sesion(
        USUARIO_INVITADO, "Ingreso como Invitado (sesión temporal)"
    )


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
        tab_login, tab_registro = st.tabs(
            ["🔑 Iniciar Sesión", "📝 Registrar Usuario"]
        )

        with tab_login:
            with st.form("form_login", clear_on_submit=False):
                usuario = st.text_input("Usuario", key="login_usuario")
                password = st.text_input(
                    "Contraseña", type="password", key="login_password"
                )
                entrar = st.form_submit_button(
                    "🚀 INICIAR SESIÓN",
                    key="btn_login",
                    use_container_width=True,
                )
            if entrar:
                if not normalizar_usuario(usuario) or not password:
                    st.error("⚠️ Ingrese su Usuario y su Contraseña.")
                elif leer_usuarios() is None:
                    st.error(
                        f"No se puede leer '{ARCHIVO_USUARIOS}' (archivo dañado)."
                    )
                else:
                    nombre = validar_credenciales(usuario, password)
                    if nombre is None:
                        st.error("⚠️ Usuario o contraseña incorrectos.")
                    else:
                        iniciar_sesion(
                            nombre, "Ingreso autorizado a la consola"
                        )
                        st.rerun()

        with tab_registro:
            with st.form("form_registro", clear_on_submit=True):
                nuevo_usuario = st.text_input
