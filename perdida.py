import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Análisis de Riesgo Crediticio", layout="wide")

# ---- Título de la aplicación ----
st.title("📋 **Risk Map Coop 360**")
st.markdown(
    "Sistema integral de análisis financiero y evaluación estadística del riesgo cooperativo")

# ================= SECCIÓN: DATOS PERSONALES ===================
st.header("📌 Datos Personales")
nombre = st.text_input("Nombre completo")
edad = st.number_input("Edad", min_value=18, max_value=99, step=1)
estado_civil = st.selectbox("Estado Civil", ["Soltero", "Casado", "Divorciado", "Viudo"])

# ================= SECCIÓN: DATOS FINANCIEROS ===================
st.header("💰 Datos Financieros")
ingreso_mensual = st.number_input("Ingreso Mensual ($)", min_value=0, step=100)
gastos_mensuales = st.number_input("Gastos Mensuales ($)", min_value=0, step=100)
deuda_actual = st.number_input("Deuda Actual Total ($)", min_value=0, step=100)
patrimonio_neto = st.number_input("Patrimonio Neto ($)", min_value=0, step=1000)
ahorros_disponibles = st.number_input("Ahorros Disponibles ($)", min_value=0, step=100)
historial_empleo = st.number_input("Años en el Empleo Actual", min_value=0, max_value=50, step=1)

# ================= SECCIÓN: HISTORIAL CREDITICIO ===================
st.header("📊 Historial Crediticio")
cuentas_credito = st.number_input("N° de Cuentas de Crédito", min_value=0, step=1)
cuentas_morosas = st.number_input("N° de Cuentas en Mora", min_value=0, max_value=cuentas_credito, step=1)
tiempo_credito = st.number_input("Años de Historial Crediticio", min_value=0, max_value=50, step=1)
pagos_atrasados = st.number_input("N° de Pagos Atrasados en el Último Año", min_value=0, max_value=12, step=1)
bancarrotas = st.number_input("N° de Bancarrotas Declaradas", min_value=0, step=1)
consultas_credito = st.number_input("N° de Consultas de Crédito Recientes", min_value=0, step=1)

# ================= SECCIÓN: DATOS DEL CRÉDITO ===================
st.header("🏦 Datos del Crédito Solicitado")
limite_credito = st.number_input("Límite Total de Crédito ($)", min_value=0, step=1000)
uso_actual_credito = st.number_input("Uso Actual del Crédito (%)", min_value=0, max_value=100, step=1)
tarjetas_credito = st.number_input("Número de Tarjetas de Crédito", min_value=0, step=1)
tipo_credito = st.selectbox("Tipo de Crédito Solicitado",
                            ["Hipotecario", "Automotriz", "Consumo", "Empresarial", "Educativo"])
monto_credito = st.number_input("Monto del Crédito Solicitado ($)", min_value=0, step=100)
plazo_credito = st.number_input("Plazo del Crédito (meses)", min_value=6, max_value=360, step=6)

# ================= PROCESO DE EVALUACIÓN ===================
if st.button("📊 Evaluar Riesgo"):

    # ---- Cálculo de PD ----
    coeficientes = [-2.5, 0.01, -0.5, 1.2, 0.8, -0.3, -0.7, 0.5, 1.1, 0.9]
    x_values = np.array([
        edad / 100, ingreso_mensual / 10000, cuentas_credito / 10,
        cuentas_morosas / 5, uso_actual_credito / 100, deuda_actual / 50000,
        tiempo_credito / 50, pagos_atrasados / 12, bancarrotas, consultas_credito / 10
    ])

    logit_pd = np.dot(coeficientes, x_values)
    pd_score = 1 / (1 + np.exp(-logit_pd))  # Transformación logística

    # ---- Cálculo de LGD ----
    if patrimonio_neto > 0:
        lgd = max(0, 0.45 + 0.2 * (deuda_actual / patrimonio_neto) - 0.1 * (ahorros_disponibles / ingreso_mensual))
    else:
        lgd = 0.85  # Si no tiene patrimonio, la LGD es alta

    # ---- Cálculo de EAD ----
    ead = monto_credito * (uso_actual_credito / 100)

    # ---- Cálculo de EL ----
    el = pd_score * lgd * ead

    # ---- Resultados ----
    st.subheader("📊 Resultados de Evaluación")
    st.markdown(f"<h1 style='text-align: center; font-size: 48px;'>📉 PD: {pd_score:.2%}</h1>", unsafe_allow_html=True)
    st.write(f"**Pérdida Dada el Default (LGD):** {lgd:.2%}")
    st.write(f"**Exposición al Default (EAD):** ${ead:,.2f}")
    st.write(f"**Pérdida Esperada (EL):** ${el:,.2f}")

    # =================== GRÁFICOS ===================
    st.subheader("📊 Análisis Financiero")

    col1, col2 = st.columns(2)

    # Gráfico de pastel - Distribución Financiera
    with col1:
        fig_pie_finanzas = px.pie(
            names=["Ingresos", "Gastos", "Deuda"],
            values=[ingreso_mensual, gastos_mensuales, deuda_actual],
            title="Distribución Financiera"
        )
        st.plotly_chart(fig_pie_finanzas)

    # Gráfico de pastel - Uso del Crédito
    with col2:
        fig_pie_credito = px.pie(
            names=["Límite de Crédito Usado", "Disponible"],
            values=[uso_actual_credito, 100 - uso_actual_credito],
            title="Uso del Crédito"
        )
        st.plotly_chart(fig_pie_credito)

    # Gráfico de barras - Comparación de Indicadores Financieros
    with col1:
        indicadores = ["Ingresos", "Gastos", "Deuda", "Ahorros", "Límite Crédito"]
        valores = [ingreso_mensual, gastos_mensuales, deuda_actual, ahorros_disponibles, limite_credito]
        fig_barras = px.bar(x=indicadores, y=valores, title="Comparación de Indicadores Financieros")
        st.plotly_chart(fig_barras)

    # Gráfico de barras - Relación Deuda/Patrimonio
    with col2:
        fig_barras_ratio = px.bar(
            x=["Deuda", "Patrimonio Neto"],
            y=[deuda_actual, patrimonio_neto],
            title="Relación Deuda vs Patrimonio"
        )
        st.plotly_chart(fig_barras_ratio)

