"""
MÉTODOS DE ORDENACIÓN Y BÚSQUEDA
Proyecto 2: Rastreo e Inventario Bovino para Exportación y Control Zoosanitario

Ejecutar con:  streamlit run app.py
Archivos necesarios en la misma carpeta: app.py, datos.py, contexto.txt, ganado_15000.csv
"""

import csv
import html
import random
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from datos import cargar_datos_ram  # noqa: E402

# ----------------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------------
ARCHIVO_CSV = "ganado_15000.csv"
ARCHIVO_ORDENADO = "ganado_15000_ordenado.csv"
BITACORA_ORDENACION = "bitacora_ordenacion.txt"
BITACORA_BUSQUEDA = "bitacora_busqueda.txt"
ARCHIVO_CONTEXTO = "contexto.txt"

COLUMNAS = [
    "siniiga_id", "peso_kg", "fecha_vacunacion", "raza_bovino", "estado_origen",
    "municipio_rancho", "estatus_exportacion", "edad_meses", "sexo_bovino",
    "nombre_rancho",
]
CAMPOS_NUMERICOS = {"siniiga_id", "peso_kg", "fecha_vacunacion", "edad_meses"}
CAMPOS_CLAVE = {1: "siniiga_id", 2: "peso_kg", 3: "fecha_vacunacion"}
ETIQUETA = {c: (c + "*" if c in CAMPOS_CLAVE.values() else c) for c in COLUMNAS}

# Counting Sort solo es viable si el rango de valores cabe en memoria.
LIMITE_RANGO_CONTEO = 2_000_000


def fmt(n):
    return f"{n:,}"


# ----------------------------------------------------------------------------
# ALGORITMOS DE ORDENACIÓN (desde cero, sin sort()/sorted())
# Retornan: (lista_ordenada, tiempo_ns, comparaciones, movimientos)
#   comparaciones = comparaciones entre claves para decidir el orden.
#   movimientos   = escrituras de registros en la estructura (asignaciones).
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
    """Método 1: Radix Sort LSD en base 10 (estable, apoyado en conteo por dígito)."""
    inicio = time.perf_counter_ns()
    comparaciones = 0
    movimientos = 0
    n = len(datos)
    if n < 2:
        return list(datos), time.perf_counter_ns() - inicio, 0, 0

    minimo, maximo = _rango(datos, campo)  # preproceso: determina # de dígitos
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
    """Método 2: Counting Sort estable (requiere rango de valores acotado)."""
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
    for reg in datos:  # distribución
        cubetas[(reg[campo] - minimo) * num_cubetas // rango].append(reg)
        movimientos += 1

    resultado = []
    for cubeta in cubetas:  # insertion sort por cubeta + concatenación
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
    return resultado, time.perf_counter_ns() - inicio, comparaciones, movimientos


def quick_sort_iterativo(datos, campo="fecha_vacunacion"):
    """Método 4: Quick Sort iterativo (pila explícita, pivote mediana de 3 al azar y
    partición en 3 vías para manejar muchas fechas repetidas)."""
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
        # pivote: mediana de 3 elementos al azar (evita el mal caso con datos ya ordenados)
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
        # Se apila primero la mayor para que la pila crezca O(log n)
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


METODOS = {
    1: ("Radix Sort", radix_sort),
    2: ("Counting Sort", counting_sort),
    3: ("Bucket Sort", bucket_sort),
    4: ("Quick Sort Iterativo", quick_sort_iterativo),
    5: ("Merge Sort", merge_sort),
}
METODOS_POR_CLAVE = {1: [1, 2], 2: [3], 3: [4, 5]}


# ----------------------------------------------------------------------------
# ALGORITMOS DE BÚSQUEDA
# Retornan: (indice_fila_1_based, registro, todos_resultados, tiempo_ns, comparaciones)
# todos_resultados = lista de (fila_1_based, registro); se arma DESPUÉS de medir el
# tiempo, para que el benchmark corresponda a localizar la primera coincidencia.
# ----------------------------------------------------------------------------
def busqueda_binaria(datos_ordenados, campo, valor_buscado):
    inicio = time.perf_counter_ns()
    comparaciones = 0
    n = len(datos_ordenados)
    lo, hi = 0, n
    while lo < hi:  # límite inferior: primera ocurrencia
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
    """Quita acentos/tildes, pasa a minúsculas y limpia espacios extra."""
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c)
    )
    return " ".join(sin_acentos.lower().strip().split())


def busqueda_lineal(datos, campo, valor_buscado):
    """Búsqueda secuencial. En campos de texto ignora mayúsculas, acentos y espacios extra;
    prefiere la coincidencia exacta y, si no existe, acepta la primera coincidencia parcial."""
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

    todos = []  # se arma fuera del cronómetro
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
# UTILIDADES (bitácoras, CSV, estado)
# ----------------------------------------------------------------------------
def escribir_bitacora(nombre_archivo, texto):
    with open(BASE_DIR / nombre_archivo, "a", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"Fecha y hora: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(texto.strip("\n") + "\n\n")


def guardar_csv_ordenado(registros):
    with open(BASE_DIR / ARCHIVO_ORDENADO, "w", newline="", encoding="utf-8-sig") as f:
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
            raise ValueError(f"El campo {campo} es numérico; '{texto}' no es un entero válido.")
    return texto


# ----------------------------------------------------------------------------
# ASIGNACIÓN CAMPO CLAVE -> ALGORITMO (Sección 5)
# ----------------------------------------------------------------------------
ALGORITMOS = {
    1: ("Radix Sort", radix_sort),
    2: ("Bucket Sort", bucket_sort),
    3: ("Quick Sort", quick_sort_iterativo),
}


# ----------------------------------------------------------------------------
# ESTADO
# ----------------------------------------------------------------------------
def iniciar_estado():
    por_defecto = dict(
        datos_originales=None, datos_actuales=None, tiempo_carga_ns=None, campo_ordenado=None,
        conteo_metodos={nombre: 0 for nombre, _ in ALGORITMOS.values()},
        reporte_orden=None, csv_guardado=False, aviso_csv=None,
        metodo_busqueda=None, campo_busqueda=None, resultado_busqueda=None, valores_campo={},
    )
    for k, v in por_defecto.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_entradas():
    """Borra lo escrito en las cajas de captura de búsqueda (vuelven a su valor inicial)."""
    for k in [k for k in st.session_state.keys() if k.startswith("valor_")]:
        del st.session_state[k]


def guardar_datos_cargados(datos, t_ns):
    """Guarda la carga en st.session_state y deja el programa en su estado inicial (sin ordenar)."""
    st.session_state.datos_originales = datos        # copia intacta, tal como salió del CSV
    st.session_state.datos_actuales = list(datos)    # lista de trabajo (se reemplaza al ordenar)
    st.session_state.tiempo_carga_ns = t_ns
    st.session_state.campo_ordenado = None
    st.session_state.conteo_metodos = {nombre: 0 for nombre, _ in ALGORITMOS.values()}
    st.session_state.reporte_orden = None
    st.session_state.aviso_csv = None
    st.session_state.metodo_busqueda = None
    st.session_state.campo_busqueda = None
    st.session_state.resultado_busqueda = None
    st.session_state.valores_campo = {}
    limpiar_entradas()


AVISO_SIN_DATOS = ("Primero debe cargar los datos en memoria RAM: presione el botón "
                   "📥 Cargar Base de Datos a Memoria RAM.")


# ----------------------------------------------------------------------------
# COMPONENTES VISUALES
# ----------------------------------------------------------------------------
def mostrar_df(df, config=None):
    try:
        st.dataframe(df, hide_index=True, width="stretch", column_config=config)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, column_config=config)


def construir_tabla(registros, reg_inicial):
    filas = []
    for i, r in enumerate(registros):
        fila = {"Reg#": reg_inicial + i}
        for c in COLUMNAS:
            fila[ETIQUETA[c]] = r[c]
        filas.append(fila)
    return pd.DataFrame(filas)


def config_columnas():
    cfg = {ETIQUETA[c]: st.column_config.NumberColumn(format="%d") for c in CAMPOS_NUMERICOS}
    cfg["Reg#"] = st.column_config.NumberColumn(format="%d")
    return cfg


def mostrar_ficha(registro, fila):
    filas = "".join(
        f"<div class='fila'><span class='campo'>{c}:</span> {html.escape(str(registro[c]))}</div>"
        for c in COLUMNAS
    )
    st.markdown(
        "<div class='ficha'><div class='ficha-titulo'>INFORMACIÓN DEL REGISTRO LOCALIZADO</div>"
        f"<div class='ficha-sub'>Ficha del Registro · Fila {fila} en memoria RAM</div>{filas}</div>",
        unsafe_allow_html=True,
    )


CSS = """
<style>
.block-container{max-width:1200px;}
.ficha{border:2px solid #2e7d32;border-radius:10px;padding:14px 18px;margin-top:10px;
       background:rgba(46,125,50,0.08);}
.ficha-titulo{font-weight:800;font-size:1.05rem;letter-spacing:.5px;border-bottom:1px solid #2e7d32;
              padding-bottom:6px;margin-bottom:2px;}
.ficha-sub{font-size:.8rem;opacity:.75;margin-bottom:8px;}
.ficha .fila{font-family:monospace;padding:2px 0;}
.ficha .campo{font-weight:700;display:inline-block;min-width:230px;}
</style>
"""


# ----------------------------------------------------------------------------
# 1. ENCABEZADO
# ----------------------------------------------------------------------------
def seccion_encabezado():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Proyecto 2: Rastreo e Inventario Bovino para Exportación y Control Zoosanitario")
    ruta = BASE_DIR / ARCHIVO_CONTEXTO
    with st.container(border=True):
        if ruta.exists():
            for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
                linea = linea.strip()
                if not linea:
                    continue
                if linea.startswith("Nota"):
                    st.caption(linea)
                else:
                    st.write(linea)
        else:
            st.warning(f"No se encontró '{ARCHIVO_CONTEXTO}' junto a app.py.")


# ----------------------------------------------------------------------------
# 2. CARGA EN RAM Y MUESTRA
# ----------------------------------------------------------------------------
def seccion_carga_y_muestra():
    st.header("Carga en RAM y muestra")
    ruta = BASE_DIR / ARCHIVO_CSV
    if not ruta.exists():
        st.warning(f"No se encontró '{ARCHIVO_CSV}' junto a app.py. Súbalo para continuar:")
        subido = st.file_uploader("Archivo CSV", type="csv")
        if subido is not None:
            ruta.write_bytes(subido.getvalue())
            st.rerun()
        return

    if st.button("📥 Cargar Base de Datos a Memoria RAM", key="btn_cargar"):
        try:
            datos, t_ns = cargar_datos_ram(str(ruta))
        except Exception as e:
            st.error(f"No se pudo cargar el archivo: {e}")
            return
        guardar_datos_cargados(datos, t_ns)

    datos = st.session_state.datos_actuales
    if datos is None:
        st.info("Los datos todavía no están en memoria RAM. Presione el botón para leer "
                f"'{ARCHIVO_CSV}' y cargarlo a la estructura.")
        return

    n = len(datos)
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros en RAM", fmt(n))
    c2.metric("Columnas", len(datos[0]))
    c3.metric("Tiempo de carga (ns)", fmt(st.session_state.tiempo_carga_ns))
    st.info(
        f"Se realizó la carga de los {fmt(n)} registros con sus {len(datos[0])} columnas a la estructura "
        f"[Lista de Diccionarios en RAM] y le tomó [{st.session_state.tiempo_carga_ns}] "
        "nanosegundos completar la acción."
    )

    campo = st.session_state.campo_ordenado
    st.caption(
        f"Orden actual en RAM: {campo}" if campo
        else "Orden actual en RAM: el original del archivo (sin ordenar). Los campos clave llevan *."
    )
    cfg = config_columnas()
    st.markdown("**Los primeros 30 registros**")
    mostrar_df(construir_tabla(datos[:30], 1), cfg)
    st.markdown("**Los últimos 30 registros**")
    ultimos = datos[-30:]
    mostrar_df(construir_tabla(ultimos, n - len(ultimos) + 1), cfg)


# ----------------------------------------------------------------------------
# 3. DISPARADOR ALEATORIO DE ORDENACIÓN
# ----------------------------------------------------------------------------
def disparar_ordenacion():
    """Callback del botón: corre ANTES de dibujar la página, así las tablas ya salen actualizadas."""
    if st.session_state.datos_originales is None:
        return  # la advertencia se muestra en la sección
    # Siempre se ordena una copia fresca de los datos originales (desordenados)
    datos = list(st.session_state.datos_originales)
    n = len(datos)
    clave = random.choice([1, 2, 3])
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
    st.session_state.reporte_orden = texto
    st.session_state.resultado_busqueda = None  # las filas de una búsqueda previa ya no son válidas
    escribir_bitacora(BITACORA_ORDENACION, texto)

    if not st.session_state.csv_guardado:
        guardar_csv_ordenado(ordenada)
        st.session_state.csv_guardado = True
        st.session_state.aviso_csv = (
            f"Primera ejecución: se guardó '{ARCHIVO_ORDENADO}' con los {fmt(n)} registros "
            f"ordenados por [{campo}]."
        )


def seccion_ordenacion():
    st.header("Disparador aleatorio de ordenación")
    st.button("🎲 Disparador Aleatorio de Ordenación", key="btn_ordenar", on_click=disparar_ordenacion)
    if st.session_state.datos_originales is None:
        st.warning(AVISO_SIN_DATOS)
    elif st.session_state.reporte_orden:
        st.success(st.session_state.reporte_orden)
        st.caption(f"Reporte agregado a {BITACORA_ORDENACION}.")
        if st.session_state.aviso_csv:
            st.info(st.session_state.aviso_csv)
    else:
        st.caption("Sortea al azar el campo clave (siniiga_id, peso_kg o fecha_vacunacion) y su algoritmo.")


# ----------------------------------------------------------------------------
# 4. PANEL DEL MÉTODO ALFA
# ----------------------------------------------------------------------------
def seccion_metodo_alfa():
    st.header("Panel del Método Alfa")
    conteo = st.session_state.conteo_metodos
    maximo = max(conteo.values())
    alfas = [nombre for nombre, v in conteo.items() if v == maximo] if maximo else []
    cols = st.columns(len(conteo))
    for col, (nombre, veces) in zip(cols, conteo.items()):
        with col:
            with st.container(border=True):
                es_alfa = nombre in alfas
                st.metric(("👑 " if es_alfa else "") + nombre, f"{veces} vez/veces")
    if not alfas:
        st.write("**Método Alfa:** aún no se ha ejecutado ninguna ordenación.")
    elif len(alfas) == 1:
        st.success(f"👑 Método Alfa actual: **{alfas[0]}** ({maximo} selección(es)).")
    else:
        st.success(f"👑 Empate de Método Alfa entre **{', '.join(alfas)}** ({maximo} selección(es) cada uno).")


# ----------------------------------------------------------------------------
# 5. DISPARADOR ALEATORIO DE BÚSQUEDA
# ----------------------------------------------------------------------------
def disparar_busqueda():
    if st.session_state.datos_originales is None:
        return  # la advertencia se muestra en la sección
    metodo = random.choice(["Búsqueda Binaria", "Búsqueda Lineal"])
    campos_disponibles = list(COLUMNAS)  # los 10 campos
    st.session_state.metodo_busqueda = metodo
    st.session_state.campo_busqueda = (
        random.choice(campos_disponibles) if metodo == "Búsqueda Lineal" else None
    )
    st.session_state.resultado_busqueda = None
    limpiar_entradas()


def ejecutar_busqueda(metodo, campo, valor, valor_txt):
    n = len(st.session_state.datos_actuales)
    datos = st.session_state.datos_actuales
    if metodo == "Búsqueda Binaria":
        fila, reg, todos, t_ns, comps = busqueda_binaria(datos, campo, valor)
        cabecera = (
            "[FASE 3: DISPARADOR ALEATORIO DE BÚSQUEDA]\n"
            "Activando generador de aleatoriedad para la búsqueda del sistema...\n"
            f"-> Método de búsqueda seleccionado aleatoriamente: [{metodo}]\n"
            "[VALIDACIÓN DE SEGURIDAD]: Verificando estado físico de la memoria...\n"
            f"El campo clave requerido para la búsqueda es: [{campo}]\n"
            "Estado del campo en memoria: YA ORDENADO PREVIAMENTE.\n"
            f"Introduzca el dato a buscar ({campo}): {valor_txt}\n"
            "Ejecutando algoritmo de Búsqueda Binaria...\n"
        )
    else:
        fila, reg, todos, t_ns, comps = busqueda_lineal(datos, campo, valor)
        cabecera = (
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
        f"\"Se realizó la búsqueda en {fmt(n)} registros por el campo [{campo}] usando el método de "
        f"búsqueda [{metodo}] en [{t_ns}] nanosegundos. Comparaciones: [{comps}]\""
    )
    texto = cabecera + resultado + benchmark
    escribir_bitacora(BITACORA_BUSQUEDA, texto)
    st.session_state.resultado_busqueda = dict(texto=texto, fila=fila, registro=reg, todos=todos)


MAX_VALORES_AYUDA = 40


def valores_disponibles(campo):
    """Valores distintos del campo en el padrón (CSV original), ordenados sin importar acentos."""
    cache = st.session_state.valores_campo
    if campo not in cache:
        vals = {str(r[campo]).strip() for r in st.session_state.datos_originales}
        clave = (lambda v: int(v)) if campo in CAMPOS_NUMERICOS else normalizar_texto
        cache[campo] = sorted((v for v in vals if v), key=clave)
    return cache[campo]


def texto_valores_disponibles(campo):
    vals = valores_disponibles(campo)
    texto = "Valores en el padrón: " + ", ".join(vals[:MAX_VALORES_AYUDA])
    if len(vals) > MAX_VALORES_AYUDA:
        texto += f" … (y {len(vals) - MAX_VALORES_AYUDA} más; {fmt(len(vals))} valores distintos en total)"
    return texto


FORMATOS_CLAVE = {
    "siniiga_id": "Formato requerido: Número entero de 8 dígitos sin espacios ni guiones "
                  "(Rango: 10000000 a 99999999).",
    "peso_kg": "Formato requerido: Número entero expresado en kilogramos sin decimales "
               "(Rango en padrón: 280 a 800).",
    "fecha_vacunacion": "Formato requerido: Número entero en formato AAAAMMDD sin guiones ni diagonales "
                        "(Rango: 20250101 a 20261015, ej. 20260415).",
}


def ejemplos_reales(campo, cantidad=5):
    """Valores que SÍ existen ahora en la lista en RAM: primero, fila 15, un cuarto, medio y último
    (si alguno se repite, se completa con filas repartidas a lo largo de la lista)."""
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
        return ("Formato requerido: Número entero de meses sin decimales "
                f"(Rango en padrón: {min(edades)} a {max(edades)}).")
    return ("Formato requerido: Texto. No distingue mayúsculas, acentos ni espacios extra "
            "y acepta coincidencias parciales.")


def guia_campo(campo):
    """Muestra el formato esperado y ejemplos reales de la RAM; devuelve los ejemplos."""
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
    st.header("Disparador aleatorio de búsqueda")
    st.button("🎲 Disparador Aleatorio de Búsqueda", key="btn_buscar_azar", on_click=disparar_busqueda)
    if st.session_state.datos_originales is None:
        st.warning(AVISO_SIN_DATOS)
        return

    metodo = st.session_state.metodo_busqueda
    if metodo is None:
        st.caption("Sortea al azar el método (Búsqueda Lineal o Binaria) y, en la lineal, también el campo.")
        return
    st.markdown(f"**Método sorteado:** {metodo}")

    if metodo == "Búsqueda Binaria":
        campo = st.session_state.campo_ordenado
        if campo is None:
            st.warning(
                "[VALIDACIÓN DE SEGURIDAD]: la Búsqueda Binaria requiere datos ordenados. "
                "Presione primero el Disparador Aleatorio de Ordenación."
            )
            return
        st.caption(f"Campo clave ordenado en RAM (seleccionado automáticamente): {campo}.")
    else:
        campo = st.session_state.campo_busqueda
        st.caption(f"Campo seleccionado aleatoriamente: {campo}.")

    ejemplos = guia_campo(campo)
    if campo not in CAMPOS_CLAVE.values():
        st.caption(texto_valores_disponibles(campo))

    if campo in CAMPOS_NUMERICOS:
        valor = st.number_input(
            f"Introduzca el dato a buscar ({campo})", min_value=0, value=int(ejemplos[0]), step=1,
            format="%d", key=f"valor_{campo}",
        )
    else:
        valor_txt = st.text_input(f"Introduzca el dato a buscar ({campo})", key=f"valor_{campo}")

    b1, b2, _ = st.columns([1, 2, 4])
    buscar = b1.button("Buscar", key="btn_buscar")
    b2.button("Limpiar / Reiniciar búsqueda", key="btn_limpiar", on_click=limpiar_busqueda)
    if buscar:
        if campo in CAMPOS_NUMERICOS:
            ejecutar_busqueda(metodo, campo, int(valor), str(int(valor)))
        else:
            try:
                valor_buscado = parse_valor(campo, valor_txt)
            except ValueError as e:
                st.error(str(e))
            else:
                ejecutar_busqueda(metodo, campo, valor_buscado, valor_txt.strip())

    res = st.session_state.resultado_busqueda
    if res:
        st.code(res["texto"], language="text")
        if res["fila"] is None:
            st.warning("No se localizó ningún registro con ese dato.")
        else:
            mostrar_ficha(res["registro"], res["fila"])
            if len(res["todos"]) > 1:
                st.info(f"Se encontraron {len(res['todos'])} registros con ese valor; "
                        "la ficha muestra la primera ocurrencia.")
                with st.expander(f"Ver todos los resultados ({len(res['todos'])})"):
                    filas = [{"Fila": f, **{c: r[c] for c in COLUMNAS}} for f, r in res["todos"]]
                    mostrar_df(pd.DataFrame(filas))


# ----------------------------------------------------------------------------
# DESCARGAS (archivos generados)
# ----------------------------------------------------------------------------
def seccion_descargas():
    generados = [(n, m) for n, m in [
        (BITACORA_ORDENACION, "text/plain"),
        (BITACORA_BUSQUEDA, "text/plain"),
        (ARCHIVO_ORDENADO, "text/csv"),
    ] if (BASE_DIR / n).exists()]
    if not generados:
        return
    st.divider()
    st.subheader("Archivos generados")
    cols = st.columns(len(generados))
    for col, (nombre, mime) in zip(cols, generados):
        col.download_button(f"⬇ {nombre}", data=(BASE_DIR / nombre).read_bytes(),
                            file_name=nombre, mime=mime, key=f"dl_{nombre}")


def main():
    st.set_page_config(page_title="Rastreo e Inventario Bovino", page_icon="🐄", layout="wide")
    iniciar_estado()
    seccion_encabezado()
    seccion_carga_y_muestra()
    st.divider()
    seccion_ordenacion()
    st.divider()
    seccion_metodo_alfa()
    st.divider()
    seccion_busqueda()
    seccion_descargas()


if __name__ == "__main__":
    main()
