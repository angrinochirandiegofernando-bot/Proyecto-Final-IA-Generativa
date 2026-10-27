"""Agente de devoluciones de EcoMarket (Proyecto Final).

Archivo generado automáticamente desde Taller_003.ipynb (celdas con la etiqueta "agente").
Para modificar el agente, edita el notebook y vuelve a ejecutarlo."""


import contextvars
import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Literal

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field


def ubicar_raiz():
    """Busca hacia arriba la carpeta del proyecto (la que contiene 'Datos')."""
    inicio = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for carpeta in [inicio, *inicio.parents]:
        if (carpeta / "Datos").is_dir():
            return carpeta
    raise FileNotFoundError("No se encontró la carpeta 'Datos' del proyecto.")


RUTA_BASE = ubicar_raiz()
RUTA_DATOS = RUTA_BASE / "Datos"
RUTA_SALIDA = RUTA_BASE / "Salida"
RUTA_ETIQUETAS = RUTA_SALIDA / "etiquetas"
RUTA_CHROMA = RUTA_BASE / "chroma_db"
RUTA_REGISTRO_ACCIONES = RUTA_SALIDA / "registro_acciones.jsonl"
RUTA_DEVOLUCIONES = RUTA_SALIDA / "devoluciones_registradas.json"
RUTA_SALIDA.mkdir(exist_ok=True)
RUTA_ETIQUETAS.mkdir(exist_ok=True)

# El .env puede estar en la raíz del proyecto o junto a este archivo
for carpeta in [RUTA_BASE, RUTA_BASE / "Script"]:
    if (carpeta / ".env").exists():
        load_dotenv(carpeta / ".env")
        break

# Recuperación (mismos valores calibrados en el Taller 2)
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "intfloat/multilingual-e5-small")
NOMBRE_COLECCION = "ecomarket_agente"
TAMANO_CHUNK = 500
SOLAPAMIENTO = 80
TOP_K = 4
UMBRAL_SIMILITUD = 0.84

# Generación
MODELO_LLM = os.getenv("MODELO_LLM", "gemini-3.5-flash-lite")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
PAUSA_ENTRE_LLAMADAS = float(os.getenv("PAUSA_ENTRE_LLAMADAS", "4"))

# Fecha del "sistema". Se fija para que las reglas de plazo den siempre el mismo resultado
# en la demostración; en producción sería date.today().
FECHA_REFERENCIA = date.fromisoformat(os.getenv("FECHA_REFERENCIA", "2026-10-26"))

# Reglas de la política de devoluciones (Datos/politica_devoluciones.pdf)
DIAS_CALENDARIO_DEVOLUCION = 30      # sección 2
DIAS_HABILES_REPORTE_DANO = 5        # sección 5
DIAS_PARA_ENTREGAR_A_TRANSPORTADORA = 7  # sección 7, paso 3
CATEGORIAS_HIGIENE = {"Belleza y salud"}  # sección 4
CATEGORIAS_PERFUME = {"Perfumería"}       # sección 4

MOTIVOS = Literal["defectuoso", "danado_en_transporte", "producto_equivocado", "cambio_de_opinion"]

MENSAJE_SIN_INFORMACION = (
    "Lo siento, no tengo la información necesaria para atender esa solicitud. "
    "Si lo deseas, puedo comunicarte con un agente humano de soporte."
)


# Cambio respecto al Taller 2: los pedidos ya no se indexan en ChromaDB. Ahora se consultan
# con una herramienta sobre el sistema de pedidos, como recomendaba la sección de limitaciones
# del Taller 2 ("en producción, el estado de un pedido debe consultarse en la base de datos
# transaccional"). El índice vectorial queda para política, inventario y preguntas frecuentes.

def cargar_politica_pdf(ruta):
    """Lee el PDF y devuelve un Document por sección, conservando sus párrafos."""
    pdf = pymupdf.open(ruta)
    bloques = []
    for pagina in pdf:
        for bloque in pagina.get_text("blocks"):
            if bloque[6] == 0:
                texto = unicodedata.normalize("NFKC", " ".join(bloque[4].split()))
                if texto:
                    bloques.append(texto)
    pdf.close()

    titulo = bloques[0]
    secciones = []
    for bloque in bloques[1:]:
        if re.match(r"^\d+\.\s", bloque) and len(bloque) < 80:
            secciones.append({"seccion": bloque, "parrafos": []})
        elif secciones:
            secciones[-1]["parrafos"].append(bloque)

    return [
        Document(
            page_content="\n\n".join(s["parrafos"]),
            metadata={"fuente": ruta.name, "tipo": "politica", "titulo": titulo, "seccion": s["seccion"]},
        )
        for s in secciones
    ]


def cargar_inventario_csv(ruta):
    """Un Document por producto, redactado como frase."""
    documentos = []
    for _, fila in pd.read_csv(ruta).iterrows():
        disponibilidad = f"{fila['stock']} unidades disponibles" if fila["stock"] > 0 else "agotado, 0 unidades disponibles"
        garantia = f"{fila['garantia_meses']} meses" if fila["garantia_meses"] > 0 else "no tiene garantía extendida"
        texto = (
            f"Producto {fila['id_producto']}: {fila['nombre']}. Categoría: {fila['categoria']}. "
            f"Descripción: {fila['descripcion']}. Material: {fila['material']}. "
            f"Precio: {fila['precio_cop']} pesos colombianos. Inventario: {disponibilidad}. "
            f"Garantía: {garantia}. Certificación: {fila['certificacion']}."
        )
        documentos.append(Document(
            page_content=texto,
            metadata={"fuente": ruta.name, "tipo": "producto", "id_producto": fila["id_producto"]},
        ))
    return documentos


def cargar_faq_json(ruta):
    """Un Document por pregunta frecuente: la pregunta y su respuesta van juntas."""
    preguntas = json.loads(ruta.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=f"Pregunta frecuente ({p['categoria']}): {p['pregunta']}\nRespuesta: {p['respuesta']}",
            metadata={"fuente": ruta.name, "tipo": "faq", "id_faq": p["id"], "categoria": p["categoria"]},
        )
        for p in preguntas
    ]


def segmentar(documentos, tamano=TAMANO_CHUNK, solapamiento=SOLAPAMIENTO):
    """Registros completos para CSV y JSON; recursiva por sección para el PDF."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamano,
        chunk_overlap=solapamiento,
        separators=["\n\n", ". ", " ", ""],
        keep_separator="end",
    )
    fragmentos = []
    for documento in documentos:
        if documento.metadata["tipo"] != "politica":
            fragmentos.append(documento)
            continue
        encabezado = f"{documento.metadata['titulo']} - {documento.metadata['seccion']}"
        for numero, parte in enumerate(divisor.split_text(documento.page_content), start=1):
            fragmentos.append(Document(
                page_content=f"{encabezado}\n{parte.strip()}",
                metadata={**documento.metadata, "parte": numero},
            ))
    return fragmentos


def identificador(fragmento):
    m = fragmento.metadata
    if m["tipo"] == "politica":
        return f"politica-s{m['seccion'].split('.')[0]}-p{m['parte']}"
    return m.get("id_producto") or m.get("id_faq")


class _BaseConocimiento:
    """Carga perezosa del modelo de embeddings y del índice: solo se construyen la primera vez que se usan."""

    def __init__(self):
        self._almacen = None

    @property
    def almacen(self):
        if self._almacen is None:
            self._almacen = self._construir()
        return self._almacen

    def _construir(self):
        import chromadb
        from langchain_chroma import Chroma
        from langchain_core.embeddings import Embeddings
        from sentence_transformers import SentenceTransformer

        class EmbeddingsE5(Embeddings):
            """Adapta un modelo E5 de sentence-transformers a la interfaz de LangChain."""

            def __init__(self, nombre_modelo):
                self.modelo = SentenceTransformer(nombre_modelo, device="cpu")

            def embed_documents(self, textos):
                return self.modelo.encode([f"passage: {t}" for t in textos], normalize_embeddings=True).tolist()

            def embed_query(self, texto):
                return self.modelo.encode(f"query: {texto}", normalize_embeddings=True).tolist()

        fragmentos = segmentar(
            cargar_politica_pdf(RUTA_DATOS / "politica_devoluciones.pdf")
            + cargar_inventario_csv(RUTA_DATOS / "inventario_productos.csv")
            + cargar_faq_json(RUTA_DATOS / "preguntas_frecuentes.json")
        )
        cliente = chromadb.PersistentClient(path=str(RUTA_CHROMA))
        almacen = Chroma(
            client=cliente,
            collection_name=NOMBRE_COLECCION,
            embedding_function=EmbeddingsE5(MODELO_EMBEDDINGS),
            collection_metadata={"hnsw:space": "cosine"},
        )
        # Los identificadores son estables: si el índice ya existe y está completo, no se recalcula
        if almacen._collection.count() != len(fragmentos):
            cliente.delete_collection(NOMBRE_COLECCION)
            almacen = Chroma(
                client=cliente,
                collection_name=NOMBRE_COLECCION,
                embedding_function=almacen._embedding_function,
                collection_metadata={"hnsw:space": "cosine"},
            )
            almacen.add_documents(fragmentos, ids=[identificador(f) for f in fragmentos])
        return almacen


base_conocimiento = _BaseConocimiento()


def recuperar(consulta, k=TOP_K, umbral=UMBRAL_SIMILITUD):
    """Búsqueda semántica con el umbral de relevancia calibrado en el Taller 2."""
    encontrados = []
    for documento, distancia in base_conocimiento.almacen.similarity_search_with_score(consulta, k=k):
        similitud = round(1 - distancia, 3)
        if similitud >= umbral:
            encontrados.append(Document(page_content=documento.page_content, metadata={**documento.metadata, "similitud": similitud}))
    return encontrados


# Simula el sistema transaccional de EcoMarket y el servicio de la transportadora.
# Las reglas son funciones de Python deterministas: la decisión de si una devolución
# procede no la toma el modelo de lenguaje sino este código, que aplica la política.

class SistemaPedidos:
    def __init__(self):
        self.pedidos = pd.read_csv(RUTA_DATOS / "pedidos_ecomarket.csv", dtype=str, keep_default_na=False)
        self.inventario = pd.read_csv(RUTA_DATOS / "inventario_productos.csv").set_index("id_producto")
        # Permite demostrar el manejo de errores: simula que la transportadora no responde
        self.simular_falla_transportadora = False

    # --- Devoluciones registradas (persisten en un JSON para evitar etiquetas duplicadas) ---

    def devoluciones(self):
        if RUTA_DEVOLUCIONES.exists():
            return json.loads(RUTA_DEVOLUCIONES.read_text(encoding="utf-8"))
        return {}

    def _guardar_devoluciones(self, devoluciones):
        RUTA_DEVOLUCIONES.write_text(json.dumps(devoluciones, ensure_ascii=False, indent=2), encoding="utf-8")

    def reiniciar_devoluciones(self):
        """Borra las devoluciones registradas y sus etiquetas (útil para repetir la demostración)."""
        if RUTA_DEVOLUCIONES.exists():
            RUTA_DEVOLUCIONES.unlink()
        for etiqueta in RUTA_ETIQUETAS.glob("*.pdf"):
            etiqueta.unlink()

    # --- Consultas ---

    def buscar(self, numero_pedido, correo_cliente):
        """Devuelve el pedido solo si el número y el correo coinciden; si no, None.

        No distingue entre "el pedido no existe" y "el correo no coincide" para no revelar
        a un tercero qué números de pedido son válidos.
        """
        numero = (numero_pedido or "").strip().upper()
        correo = (correo_cliente or "").strip().lower()
        fila = self.pedidos[(self.pedidos["numero_pedido"] == numero) & (self.pedidos["correo_cliente"] == correo)]
        if fila.empty:
            return None
        pedido = fila.iloc[0].to_dict()
        producto = self.inventario.loc[pedido["id_producto"]]
        pedido.update({
            "nombre_producto": producto["nombre"],
            "categoria": producto["categoria"],
            "garantia_meses": int(producto["garantia_meses"]),
        })
        return pedido

    @staticmethod
    def resumen(pedido):
        return {
            "numero_pedido": pedido["numero_pedido"],
            "producto": f"{pedido['nombre_producto']} ({pedido['id_producto']})",
            "cantidad": int(pedido["cantidad"]),
            "estado": pedido["estado"],
            "fecha_compra": pedido["fecha_compra"],
            "fecha_entrega": pedido["fecha_entrega"] or "aún no entregado",
            "ciudad_destino": pedido["ciudad_destino"],
        }

    # --- Reglas de elegibilidad ---

    def evaluar_elegibilidad(self, numero_pedido, correo_cliente, motivo,
                             producto_abierto=False, producto_usado=False):
        """Aplica la política de devoluciones y devuelve la decisión con su justificación."""
        if motivo not in MOTIVOS.__args__:
            return {"ok": False, "elegible": False, "codigo": "MOTIVO_NO_VALIDO",
                    "detalle": f"El motivo debe ser uno de: {', '.join(MOTIVOS.__args__)}."}

        pedido = self.buscar(numero_pedido, correo_cliente)
        if pedido is None:
            return {"ok": False, "elegible": False, "codigo": "PEDIDO_NO_ENCONTRADO",
                    "detalle": "No hay ningún pedido con ese número asociado a ese correo electrónico."}

        def decision(elegible, codigo, detalle, seccion, condiciones=None):
            return {"ok": True, "elegible": elegible, "codigo": codigo, "detalle": detalle,
                    "seccion_politica": seccion, "pedido": self.resumen(pedido), "condiciones": condiciones}

        registrada = self.devoluciones().get(pedido["numero_pedido"])
        if registrada:
            return decision(False, "DEVOLUCION_YA_REGISTRADA",
                            f"Este pedido ya tiene una devolución registrada con la etiqueta {registrada['id_etiqueta']} "
                            f"del {registrada['fecha_generacion']}.", "7. Cómo solicitar una devolución")

        if pedido["estado"] == "Cancelado":
            return decision(False, "PEDIDO_CANCELADO",
                            "El pedido fue cancelado, así que no hay producto que devolver. El reembolso de un pedido "
                            "cancelado antes del envío es total y se refleja entre 3 y 5 días hábiles.", "8. Reembolsos")
        if pedido["estado"] != "Entregado" or not pedido["fecha_entrega"]:
            return decision(False, "PEDIDO_NO_ENTREGADO",
                            f"El pedido está en estado '{pedido['estado']}' y todavía no ha sido entregado. "
                            "La devolución solo se puede solicitar después de recibir el producto.",
                            "2. Plazo para solicitar una devolución")

        entrega = date.fromisoformat(pedido["fecha_entrega"])
        dias_calendario = (FECHA_REFERENCIA - entrega).days
        dias_habiles = dias_habiles_entre(entrega, FECHA_REFERENCIA)
        valor = int(pedido["valor_producto_cop"])
        envio = int(pedido["costo_envio_cop"])

        if motivo in ("defectuoso", "danado_en_transporte"):
            if dias_habiles <= DIAS_HABILES_REPORTE_DANO:
                return decision(True, "ELEGIBLE",
                                f"El producto se recibió hace {dias_habiles} días hábiles y los daños o defectos se pueden "
                                f"reportar hasta {DIAS_HABILES_REPORTE_DANO} días hábiles después de la entrega. "
                                "Las exclusiones de higiene y perfumes no aplican a productos defectuosos o dañados.",
                                "5. Productos defectuosos o dañados",
                                {"costo_envio_retorno": "Lo asume EcoMarket",
                                 "opciones": f"Reembolso total de {valor + envio} pesos (producto más envío original) "
                                             "o reemplazo del producto sin costo",
                                 "requisitos": ["Adjuntar al menos dos fotografías del producto y del empaque"]})
            meses_desde_entrega = dias_calendario / 30.4
            if pedido["garantia_meses"] and meses_desde_entrega <= pedido["garantia_meses"]:
                return decision(False, "APLICA_GARANTIA",
                                f"Pasaron {dias_habiles} días hábiles desde la entrega, más que los "
                                f"{DIAS_HABILES_REPORTE_DANO} permitidos para reportar daños como devolución. El producto "
                                f"sigue cubierto por su garantía de {pedido['garantia_meses']} meses por defectos de "
                                "fabricación, que gestiona un agente humano de soporte.", "9. Garantía")
            return decision(False, "PLAZO_VENCIDO",
                            f"Pasaron {dias_habiles} días hábiles desde la entrega y el plazo para reportar daños es de "
                            f"{DIAS_HABILES_REPORTE_DANO} días hábiles. El producto no tiene garantía vigente.",
                            "5. Productos defectuosos o dañados")

        if dias_calendario > DIAS_CALENDARIO_DEVOLUCION:
            return decision(False, "PLAZO_VENCIDO",
                            f"El producto se recibió hace {dias_calendario} días y el plazo para solicitar una devolución "
                            f"es de {DIAS_CALENDARIO_DEVOLUCION} días calendario.", "2. Plazo para solicitar una devolución")

        if motivo == "producto_equivocado":
            return decision(True, "ELEGIBLE",
                            f"El producto no corresponde a lo pedido y se recibió hace {dias_calendario} días, dentro del "
                            f"plazo de {DIAS_CALENDARIO_DEVOLUCION} días.", "3. Productos que sí se pueden devolver",
                            {"costo_envio_retorno": "Lo asume EcoMarket (error en el despacho)",
                             "opciones": f"Reembolso total de {valor + envio} pesos o envío del producto correcto",
                             "requisitos": ["Devolver el producto recibido con su empaque"]})

        # motivo == "cambio_de_opinion"
        if pedido["categoria"] in CATEGORIAS_HIGIENE and producto_abierto:
            return decision(False, "EXCLUSION_HIGIENE",
                            "Por razones sanitarias, los productos de higiene personal que fueron abiertos después de la "
                            "entrega no se pueden devolver.", "4. Productos que no se pueden devolver")
        if pedido["categoria"] in CATEGORIAS_PERFUME and (producto_abierto or producto_usado):
            return decision(False, "EXCLUSION_PERFUME",
                            "Los perfumes que fueron abiertos o utilizados no se pueden devolver.",
                            "4. Productos que no se pueden devolver")
        if producto_usado:
            return decision(False, "PRODUCTO_USADO",
                            "En una devolución por cambio de opinión el producto debe estar sin usar, en su empaque "
                            "original y con todos sus accesorios.", "3. Productos que sí se pueden devolver")
        return decision(True, "ELEGIBLE",
                        f"El producto se recibió hace {dias_calendario} días, dentro del plazo de "
                        f"{DIAS_CALENDARIO_DEVOLUCION} días, y está sin usar.", "6. Devolución por cambio de opinión",
                        {"costo_envio_retorno": "Lo paga el cliente",
                         "opciones": f"Reembolso de {valor} pesos (solo el precio del producto; el envío original no se "
                                     "reembolsa) o cambio por otro producto de igual o mayor valor",
                         "requisitos": ["Producto sin usar, en su empaque original y con todos sus accesorios"]})

    # --- Generación de la etiqueta ---

    def generar_etiqueta(self, numero_pedido, correo_cliente, motivo, producto_abierto=False, producto_usado=False):
        """Genera la etiqueta solo si la devolución es elegible. Vuelve a verificar: no confía en el agente."""
        evaluacion = self.evaluar_elegibilidad(numero_pedido, correo_cliente, motivo, producto_abierto, producto_usado)
        if not evaluacion["elegible"]:
            return {"ok": False, "codigo": "NO_ELEGIBLE", "motivo_rechazo": evaluacion["codigo"],
                    "detalle": evaluacion["detalle"]}

        if self.simular_falla_transportadora:
            return {"ok": False, "codigo": "TRANSPORTADORA_NO_DISPONIBLE",
                    "detalle": "El servicio de la transportadora no respondió. La devolución es elegible pero la "
                               "etiqueta no se pudo generar; no se registró ninguna devolución."}

        devoluciones = self.devoluciones()
        pedido = evaluacion["pedido"]
        id_etiqueta = f"DEV-{FECHA_REFERENCIA:%Y%m%d}-{len(devoluciones) + 1:04d}"
        paga_ecomarket = motivo != "cambio_de_opinion"
        etiqueta = {
            "id_etiqueta": id_etiqueta,
            "numero_guia": "ECOENV" + hashlib.sha1(f"{id_etiqueta}{pedido['numero_pedido']}".encode()).hexdigest()[:10].upper(),
            "transportadora": "EcoEnvíos (simulada)",
            "numero_pedido": pedido["numero_pedido"],
            "producto": pedido["producto"],
            "motivo": motivo,
            "remitente_ciudad": pedido["ciudad_destino"],
            "destino": "Bodega de devoluciones EcoMarket, Yumbo (Valle del Cauca)",
            "envio_pagado_por": "EcoMarket" if paga_ecomarket else "Cliente (contra entrega del retorno)",
            "fecha_generacion": FECHA_REFERENCIA.isoformat(),
            "entregar_a_transportadora_antes_de": (FECHA_REFERENCIA + timedelta(days=DIAS_PARA_ENTREGAR_A_TRANSPORTADORA)).isoformat(),
            "condiciones": evaluacion["condiciones"],
        }
        etiqueta["archivo"] = str(escribir_etiqueta_pdf(etiqueta).relative_to(RUTA_BASE))
        devoluciones[pedido["numero_pedido"]] = etiqueta
        self._guardar_devoluciones(devoluciones)
        return {"ok": True, "codigo": "ETIQUETA_GENERADA", **etiqueta}


def dias_habiles_entre(inicio, fin):
    """Días hábiles (lunes a viernes) transcurridos después de 'inicio' hasta 'fin', incluido. No considera festivos."""
    return sum(1 for n in range(1, (fin - inicio).days + 1) if (inicio + timedelta(days=n)).weekday() < 5)


def escribir_etiqueta_pdf(etiqueta):
    ruta = RUTA_ETIQUETAS / f"{etiqueta['id_etiqueta']}.pdf"
    filas = "".join(
        f"<tr><td><b>{campo}</b></td><td>{valor}</td></tr>"
        for campo, valor in [
            ("Etiqueta", etiqueta["id_etiqueta"]),
            ("Guía", etiqueta["numero_guia"]),
            ("Transportadora", etiqueta["transportadora"]),
            ("Pedido", etiqueta["numero_pedido"]),
            ("Producto", etiqueta["producto"]),
            ("Motivo", etiqueta["motivo"].replace("_", " ")),
            ("Origen", etiqueta["remitente_ciudad"]),
            ("Destino", etiqueta["destino"]),
            ("Envío pagado por", etiqueta["envio_pagado_por"]),
            ("Entregar antes de", etiqueta["entregar_a_transportadora_antes_de"]),
        ]
    )
    html = (f"<h1>EcoMarket - Etiqueta de devolución</h1><p>Generada el {etiqueta['fecha_generacion']}. "
            f"Documento de prueba de un sistema simulado.</p><table>{filas}</table>")
    css = "body{font-family:sans-serif;font-size:11pt} h1{font-size:16pt} td{padding:4pt 10pt;border-bottom:1px solid #999}"
    historia = pymupdf.Story(html=html, user_css=css)
    pagina = pymupdf.Rect(0, 0, 420, 400)
    escritor = pymupdf.DocumentWriter(str(ruta))
    dispositivo = escritor.begin_page(pagina)
    historia.place(pagina + (24, 24, -24, -24))
    historia.draw(dispositivo)
    escritor.end_page()
    escritor.close()
    return ruta


sistema = SistemaPedidos()


# Cada llamada a una herramienta queda en Salida/registro_acciones.jsonl con la sesión,
# los argumentos (correo enmascarado), el código de resultado y la duración.

sesion_actual = contextvars.ContextVar("sesion_actual", default="local")


def enmascarar_correo(correo):
    if not isinstance(correo, str) or "@" not in correo:
        return correo
    usuario, dominio = correo.split("@", 1)
    return f"{usuario[:1]}***@{dominio}"


def registrar(evento):
    evento = {"fecha": datetime.now().isoformat(timespec="seconds"), "sesion": sesion_actual.get(), **evento}
    with RUTA_REGISTRO_ACCIONES.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def con_registro(funcion):
    """Registra la llamada y convierte cualquier excepción en una respuesta de error que el agente pueda explicar."""
    @wraps(funcion)
    def envoltura(**argumentos):
        inicio = time.perf_counter()
        try:
            resultado = funcion(**argumentos)
        except Exception as error:  # una herramienta nunca debe tumbar al agente
            resultado = {"ok": False, "codigo": "ERROR_INTERNO", "detalle": f"Error del sistema: {type(error).__name__}"}
        registrar({
            "herramienta": funcion.__name__,
            "argumentos": {k: enmascarar_correo(v) if k == "correo_cliente" else v for k, v in argumentos.items()},
            "ok": resultado.get("ok"),
            "codigo": resultado.get("codigo"),
            "duracion_ms": round((time.perf_counter() - inicio) * 1000),
        })
        return json.dumps(resultado, ensure_ascii=False)
    return envoltura


def leer_registro(ultimas=None):
    if not RUTA_REGISTRO_ACCIONES.exists():
        return pd.DataFrame()
    eventos = [json.loads(linea) for linea in RUTA_REGISTRO_ACCIONES.read_text(encoding="utf-8").splitlines() if linea]
    return pd.DataFrame(eventos[-ultimas:] if ultimas else eventos)


# El nombre, la descripción y el esquema de argumentos es lo único que el modelo ve de cada
# herramienta: de ellos depende que sepa cuándo usarla y con qué datos.

class EntradaBusqueda(BaseModel):
    consulta: str = Field(description="Pregunta o tema a buscar, redactado en español")


class EntradaPedido(BaseModel):
    numero_pedido: str = Field(description="Número del pedido, con formato EM-XXXX (por ejemplo EM-1001)")
    correo_cliente: str = Field(description=(
        "Correo electrónico con el que el cliente hizo la compra, tal como lo escribió el cliente en la conversación. "
        "Si el cliente no lo ha escrito, no llames la herramienta: pídeselo. Nunca lo inventes ni lo completes"))


class EntradaDevolucion(EntradaPedido):
    motivo: MOTIVOS = Field(description=(
        "Motivo de la devolución: 'defectuoso' (no funciona o tiene fallas), 'danado_en_transporte' (llegó roto o "
        "golpeado), 'producto_equivocado' (no es lo que pidió) o 'cambio_de_opinion' (no le gustó o ya no lo quiere)"))
    producto_abierto: bool = Field(default=False, description="True si el cliente abrió el empaque o rompió el sello")
    producto_usado: bool = Field(default=False, description="True si el cliente ya usó el producto")


@tool("buscar_informacion_ecomarket", args_schema=EntradaBusqueda)
@con_registro
def buscar_informacion_ecomarket(consulta):
    """Busca en la base de conocimiento de EcoMarket (política de devoluciones, reembolsos y garantías, catálogo de
    productos con precios y existencias, y preguntas frecuentes sobre envíos, pagos y cuenta). Úsala para cualquier
    pregunta general sobre EcoMarket. No sirve para consultar pedidos concretos."""
    fragmentos = recuperar(consulta)
    if not fragmentos:
        return {"ok": True, "codigo": "SIN_RESULTADOS",
                "detalle": "Ningún documento de EcoMarket es relevante para esta consulta."}
    return {"ok": True, "codigo": "RESULTADOS", "fragmentos": [
        {"fuente": f.metadata["fuente"], "similitud": f.metadata["similitud"], "texto": f.page_content} for f in fragmentos
    ]}


@tool("consultar_pedido", args_schema=EntradaPedido)
@con_registro
def consultar_pedido(numero_pedido, correo_cliente):
    """Consulta en el sistema de pedidos el estado, el producto y las fechas de un pedido. Requiere el número de
    pedido y el correo del cliente, que deben coincidir."""
    pedido = sistema.buscar(numero_pedido, correo_cliente)
    if pedido is None:
        return {"ok": False, "codigo": "PEDIDO_NO_ENCONTRADO",
                "detalle": "No hay ningún pedido con ese número asociado a ese correo electrónico."}
    return {"ok": True, "codigo": "PEDIDO_ENCONTRADO", "pedido": sistema.resumen(pedido)}


@tool("verificar_elegibilidad_devolucion", args_schema=EntradaDevolucion)
@con_registro
def verificar_elegibilidad_devolucion(numero_pedido, correo_cliente, motivo, producto_abierto=False, producto_usado=False):
    """Verifica si un pedido puede devolverse según la política de EcoMarket (plazos, exclusiones de higiene y
    perfumes, estado del pedido, devoluciones ya registradas). Devuelve si es elegible, el motivo, la sección de la
    política aplicada y, si procede, quién paga el envío y el reembolso. No genera ninguna etiqueta."""
    return sistema.evaluar_elegibilidad(numero_pedido, correo_cliente, motivo, producto_abierto, producto_usado)


@tool("generar_etiqueta_devolucion", args_schema=EntradaDevolucion)
@con_registro
def generar_etiqueta_devolucion(numero_pedido, correo_cliente, motivo, producto_abierto=False, producto_usado=False):
    """Genera y registra la etiqueta de envío de una devolución. Úsala solo después de que
    verificar_elegibilidad_devolucion haya indicado que es elegible y el cliente haya confirmado que quiere
    continuar. Esta acción registra la devolución en el sistema y no se puede repetir para el mismo pedido."""
    return sistema.generar_etiqueta(numero_pedido, correo_cliente, motivo, producto_abierto, producto_usado)


HERRAMIENTAS = [buscar_informacion_ecomarket, consultar_pedido, verificar_elegibilidad_devolucion, generar_etiqueta_devolucion]


PROMPT_SISTEMA = f"""Eres el asistente virtual de atención al cliente de EcoMarket, una tienda en línea de productos sostenibles en Colombia. La fecha de hoy es {FECHA_REFERENCIA.isoformat()}.

Tienes herramientas para consultar información y para actuar. Decide en cada mensaje si necesitas una:

- Preguntas generales sobre políticas, productos, precios, envíos o pagos: usa buscar_informacion_ecomarket y responde solo con lo que devuelva. Si devuelve SIN_RESULTADOS o los fragmentos no contienen la respuesta, responde: "{MENSAJE_SIN_INFORMACION}"
- Estado de un pedido: usa consultar_pedido.
- Solicitud de devolución, en este orden:
  1. Reúne el número de pedido, el correo de la compra y el motivo. Si el motivo es cambio de opinión, pregunta también si el producto fue abierto o usado. Si falta un dato, pídelo; no lo supongas.
  2. Llama a verificar_elegibilidad_devolucion.
  3. Si no es elegible, explica el motivo con amabilidad, cita la sección de la política y ofrece la alternativa que corresponda (garantía, cancelación, agente humano). No generes la etiqueta.
  4. Si es elegible, explica las condiciones (quién paga el envío, reembolso, requisitos) y pregunta si desea generar la etiqueta.
  5. Solo cuando el cliente confirme, llama a generar_etiqueta_devolucion y entrégale el número de etiqueta, la guía y la fecha límite para entregar el paquete.
- Saludos, agradecimientos o preguntas sobre lo que puedes hacer: responde directamente, sin herramientas.
- Temas ajenos a EcoMarket: responde "{MENSAJE_SIN_INFORMACION}"

REGLAS:
1. No inventes datos: pedidos, plazos, precios, montos y decisiones salen siempre de las herramientas.
2. No apruebes excepciones a la política ni prometas una devolución que la herramienta rechazó.
3. Si una herramienta devuelve un error (ok: false), explícale al cliente qué pasó en lenguaje sencillo y qué puede hacer; nunca muestres códigos internos ni JSON.
4. No reveles datos de un pedido si el número y el correo no coinciden.
5. Nunca inventes, supongas ni completes el número de pedido o el correo: si el cliente no los ha escrito en la conversación, pídeselos antes de llamar cualquier herramienta.
6. Solo puedes hacer lo que hacen tus herramientas. No tienes una herramienta para contactar a un agente humano: ofrécele al cliente esa opción y explícale cómo pedirla, pero no digas que ya lo contactaste.
7. No muestres rutas de archivos ni nombres técnicos; la aplicación le ofrece al cliente el botón para descargar la etiqueta.
8. Responde en español, con tono amable, claro y breve (máximo seis frases o una lista corta)."""


def crear_llm(api_key=None):
    """Gemini con su integración nativa; cualquier otro proveedor con el cliente compatible con OpenAI del Taller 2.

    Gemini 3 devuelve una firma de razonamiento (thought_signature) en cada llamada a herramienta y exige
    recibirla de vuelta en el siguiente turno. ChatOpenAI la descarta y Gemini responde con error 400;
    ChatGoogleGenerativeAI la conserva. Gemini 3 usa sus propios valores de muestreo, así que no se fija temperature.
    """
    api_key = api_key or LLM_API_KEY
    if not api_key:
        raise RuntimeError("Falta LLM_API_KEY. Copia .env.example como .env y escribe ahí tu clave.")
    if "generativelanguage.googleapis.com" in LLM_BASE_URL:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=MODELO_LLM, google_api_key=api_key, max_retries=6, timeout=120)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODELO_LLM, base_url=LLM_BASE_URL, api_key=api_key,
                      temperature=0, max_retries=6, timeout=120)


def texto_de(mensaje):
    """Texto de un mensaje del modelo (algunos proveedores devuelven el contenido como lista de partes)."""
    contenido = mensaje.content
    if isinstance(contenido, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in contenido).strip()
    return (contenido or "").strip()


def exigir_correo_del_cliente(request, handler):
    """Bloquea una herramienta si el correo que envía el modelo no lo escribió el cliente en la conversación.

    Evita que el modelo invente o complete el correo para avanzar sin pedírselo al cliente.
    """
    correo = str(request.tool_call["args"].get("correo_cliente", "")).strip().lower()
    if correo:
        escrito_por_cliente = " ".join(texto_de(m).lower() for m in request.state["messages"] if isinstance(m, HumanMessage))
        if correo not in escrito_por_cliente:
            registrar({"herramienta": request.tool_call["name"], "argumentos": {"correo_cliente": enmascarar_correo(correo)},
                       "ok": False, "codigo": "CORREO_NO_PROPORCIONADO", "duracion_ms": 0})
            return ToolMessage(
                content=json.dumps({"ok": False, "codigo": "CORREO_NO_PROPORCIONADO",
                                    "detalle": "El cliente no ha escrito ese correo en la conversación. Pídeselo; "
                                               "no lo inventes ni lo supongas."}, ensure_ascii=False),
                tool_call_id=request.tool_call["id"])
    return handler(request)


def crear_agente(llm=None):
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_tool_call

    return create_agent(model=llm or crear_llm(), tools=HERRAMIENTAS, system_prompt=PROMPT_SISTEMA,
                        middleware=[wrap_tool_call(exigir_correo_del_cliente)])


def responder(agente, mensaje, historial=None, sesion="local"):
    """Ejecuta un turno de conversación.

    Devuelve la respuesta, los pasos del agente (herramientas llamadas, argumentos y resultados)
    y el historial actualizado para el siguiente turno.
    """
    sesion_actual.set(sesion)
    mensajes = list(historial or []) + [HumanMessage(mensaje)]
    try:
        estado = agente.invoke({"messages": mensajes}, config={"recursion_limit": 16})
    except Exception as error:
        registrar({"herramienta": None, "ok": False, "codigo": "ERROR_MODELO", "detalle": f"{type(error).__name__}: {error}"[:300]})
        respuesta = ("Lo siento, tuve un problema técnico para procesar tu solicitud. "
                     "Por favor intenta de nuevo en un momento o pide hablar con un agente humano.")
        return {"respuesta": respuesta, "pasos": [], "historial": mensajes + [AIMessage(respuesta)], "error": str(error)}

    nuevos = estado["messages"][len(mensajes):]
    resultados = {m.tool_call_id: m.content for m in nuevos if isinstance(m, ToolMessage)}
    pasos = []
    for m in nuevos:
        if isinstance(m, AIMessage):
            for llamada in m.tool_calls:
                try:
                    resultado = json.loads(resultados.get(llamada["id"], "{}"))
                except json.JSONDecodeError:
                    resultado = {"texto": resultados.get(llamada["id"])}
                pasos.append({"herramienta": llamada["name"], "argumentos": llamada["args"], "resultado": resultado})
    respuesta = texto_de(estado["messages"][-1]) or MENSAJE_SIN_INFORMACION
    return {"respuesta": respuesta, "pasos": pasos, "historial": estado["messages"], "error": None}
