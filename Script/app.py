"""Interfaz web del agente de devoluciones de EcoMarket (Fase 4).

Ejecutar desde la raíz del proyecto:
    streamlit run Script/app.py
"""

import uuid

import streamlit as st

import agente as ag

st.set_page_config(page_title="EcoMarket - Asistente", page_icon="🌱", layout="wide")

NOMBRES_HERRAMIENTAS = {
    "buscar_informacion_ecomarket": "🔎 Búsqueda en la base de conocimiento (RAG)",
    "consultar_pedido": "📦 Consulta de pedido",
    "verificar_elegibilidad_devolucion": "✅ Verificación de elegibilidad",
    "generar_etiqueta_devolucion": "🏷️ Generación de etiqueta",
}

EJEMPLOS = [
    "Hola, ¿qué puedes hacer por mí?",
    "¿Cuántos días tengo para devolver un producto?",
    "Quiero devolver mi pedido EM-1001, la botella llegó con una abolladura. Mi correo es ana.gomez@correo.com",
    "Quiero devolver el perfume del pedido EM-1002, lo abrí y no me gustó. Correo carlos.ruiz@correo.com",
    "Mi cargador solar del pedido EM-1003 dejó de funcionar. Correo lucia.mora@correo.com",
    "¿Cuál es la capital de Francia?",
]


@st.cache_resource(show_spinner="Cargando el modelo de embeddings y el índice vectorial...")
def obtener_agente():
    ag.base_conocimiento.almacen  # construye el índice una sola vez
    return ag.crear_agente()


def mostrar_pasos(pasos):
    if not pasos:
        return
    with st.expander(f"Cómo lo resolvió el agente ({len(pasos)} herramienta{'s' if len(pasos) != 1 else ''})"):
        for paso in pasos:
            resultado = paso["resultado"]
            icono = "🟢" if resultado.get("ok") else "🔴"
            st.markdown(f"{icono} **{NOMBRES_HERRAMIENTAS.get(paso['herramienta'], paso['herramienta'])}** "
                        f"— resultado: `{resultado.get('codigo', '')}`")
            argumentos = {k: ag.enmascarar_correo(v) if k == "correo_cliente" else v for k, v in paso["argumentos"].items()}
            st.json({"argumentos": argumentos, "resultado": resultado}, expanded=False)


def mostrar_descarga(pasos, clave):
    for paso in pasos:
        resultado = paso["resultado"]
        if paso["herramienta"] == "generar_etiqueta_devolucion" and resultado.get("ok"):
            ruta = ag.RUTA_BASE / resultado["archivo"]
            if ruta.exists():
                st.download_button(f"Descargar etiqueta {resultado['id_etiqueta']} (PDF)", ruta.read_bytes(),
                                   file_name=ruta.name, mime="application/pdf", key=f"descarga-{clave}")


# --- Estado de la sesión ---
if "sesion" not in st.session_state:
    st.session_state.sesion = uuid.uuid4().hex[:8]
    st.session_state.historial = []   # mensajes de LangChain (incluye llamadas a herramientas)
    st.session_state.chat = []        # lo que se muestra: (rol, texto, pasos)

# --- Barra lateral ---
with st.sidebar:
    st.header("Panel de demostración")
    if st.button("🗨️ Nueva conversación", width="stretch"):
        for clave in ["sesion", "historial", "chat"]:
            st.session_state.pop(clave, None)
        st.rerun()

    st.subheader("Mensajes de ejemplo")
    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, width="stretch"):
            st.session_state.pendiente = ejemplo

    st.subheader("Pedidos de prueba")
    st.dataframe(ag.sistema.pedidos[["numero_pedido", "correo_cliente", "estado", "fecha_entrega"]],
                 hide_index=True, width="stretch")

    st.subheader("Simulación de fallas")
    ag.sistema.simular_falla_transportadora = st.toggle(
        "Transportadora fuera de servicio", help="La generación de etiquetas falla, para ver cómo responde el agente.")
    if st.button("Borrar devoluciones registradas", width="stretch"):
        ag.sistema.reiniciar_devoluciones()
        st.success("Devoluciones de prueba borradas.")

    st.subheader("Registro de acciones")
    registro = ag.leer_registro(ultimas=8)
    if registro.empty:
        st.caption("Aún no hay acciones registradas.")
    else:
        st.dataframe(registro[["fecha", "herramienta", "codigo", "duracion_ms"]].iloc[::-1],
                     hide_index=True, width="stretch")
    st.caption(f"Modelo: {ag.MODELO_LLM} · Fecha del sistema: {ag.FECHA_REFERENCIA} · Sesión: {st.session_state.sesion}")

# --- Conversación ---
st.title("🌱 EcoMarket · Asistente de atención al cliente")
st.caption("Pregunta por políticas, productos o envíos, consulta un pedido o solicita una devolución. "
           "El asistente verifica la elegibilidad y genera la etiqueta de envío.")

agente = obtener_agente()

for i, (rol, texto, pasos) in enumerate(st.session_state.chat):
    with st.chat_message(rol):
        st.markdown(texto)
        if rol == "assistant":
            mostrar_pasos(pasos)
            mostrar_descarga(pasos, i)

mensaje = st.chat_input("Escribe tu mensaje...") or st.session_state.pop("pendiente", None)
if mensaje:
    with st.chat_message("user"):
        st.markdown(mensaje)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            salida = ag.responder(agente, mensaje, st.session_state.historial, sesion=st.session_state.sesion)
        st.markdown(salida["respuesta"])
        mostrar_pasos(salida["pasos"])
        mostrar_descarga(salida["pasos"], len(st.session_state.chat) + 1)
    st.session_state.historial = salida["historial"]
    st.session_state.chat += [("user", mensaje, []), ("assistant", salida["respuesta"], salida["pasos"])]
    st.rerun()
