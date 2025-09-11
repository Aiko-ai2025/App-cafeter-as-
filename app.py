import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from collections import Counter
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Informe Cafetería", page_icon="☕", layout="wide")

# ===== Estilo y logo =====
LOGO_URL = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPOSITORIO/main/logo.png"  # cambia si quieres

st.markdown(f"""
<style>
    .main {{ background-color: #FAF8F2; }}
    h1, h2, h3, h4 {{ color: #0F8A84; }}
    .stButton>button {{ background-color: #0F8A84; color: white; border-radius: 8px; }}
    header {{ visibility: hidden; }}
    [data-testid="stSidebar"] {{ background-color: #FAF8F2; }}
    .logo-container {{ display: flex; justify-content: center; margin-bottom: 10px; }}
</style>
<div class="logo-container">
    <img src="{LOGO_URL}" width="150">
</div>
""", unsafe_allow_html=True)

st.title("☕ Análisis Inteligente para Cafeterías")
st.markdown("""
Sube tu archivo de ventas mensual para analizar automáticamente el rendimiento de tu cafetería.

La app analiza:
- KPIs clave y alertas
- Gráficos interactivos
- Simulación de precios con IA numérica local
- Recomendaciones de combos
- PDF descargable
""")

# ===== Utilidades =====
ALIAS = {
    "fecha": ["fecha","date","dia","día"],
    "hora": ["hora","time"],
    "ticket_id": ["ticket_id","ticket","id_ticket","ticketid"],
    "producto": ["producto","articulo","artículo","item","product"],
    "categoria": ["categoria","categoría","category","familia","grupo"],
    "cantidad": ["cantidad","unidades","uds","qty","quantity"],
    "precio_unitario": ["precio_unitario","precio","pvp","unit_price","importe_unit"],
    "coste_unitario": ["coste_unitario","costo_unitario","cost","unit_cost"],
    "total": ["total","importe_total","amount","subtotal","importe"]
}
DOW_MAP = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
DOW_ORDER = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for std, aliases in ALIAS.items():
        if std in [c.lower() for c in df.columns]:
            continue
        for a in aliases:
            if a in cols_lower:
                mapping[cols_lower[a]] = std
                break
    df = df.rename(columns=mapping)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file)
    # CSV con coma o ;
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";")

def eur(x: float) -> str:
    try:
        return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{x:.2f} €"

# ===== Carga de archivo =====
uploaded_file = st.file_uploader("Sube tu archivo Excel o CSV:", type=["xlsx","xls","csv"])
if not uploaded_file:
    st.info("Sube un archivo para empezar.")
    st.stop()

try:
    df_raw = load_any(uploaded_file)
    df = map_columns(df_raw)
    # Requisitos mínimos
    if "fecha" not in df.columns or "producto" not in df.columns:
        st.error("Faltan columnas obligatorias: 'Fecha' y 'Producto' (o sus equivalentes).")
        st.stop()

    # Tipos y limpieza
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "hora" in df.columns:
        df["hora"] = df["hora"].astype(str).str[:5]
    else:
        df["hora"] = "09:00"  # valor por defecto si no hay hora

    df["cantidad"] = pd.to_numeric(df.get("cantidad", 1), errors="coerce").fillna(1)
    if "precio_unitario" in df.columns:
        df["precio_unitario"] = pd.to_numeric(df["precio_unitario"], errors="coerce")
    if "coste_unitario" in df.columns:
        df["coste_unitario"] = pd.to_numeric(df["coste_unitario"], errors="coerce").fillna(0.0)
    else:
        df["coste_unitario"] = 0.0

    # Total si falta
    if "total" not in df.columns or df["total"].isna().all():
        if "precio_unitario" in df.columns:
            df["total"] = (df["cantidad"] * df["precio_unitario"]).round(2)
        else:
            st.error("No hay 'Total' ni 'Precio_unitario' para calcular importes.")
            st.stop()

    df = df.dropna(subset=["fecha","producto"])
    df = df[df["cantidad"] > 0]
    if "precio_unitario" in df.columns:
        df = df[df["precio_unitario"] >= 0]

    # Ticket: real si existe, si no usar Fecha+Hora
    if "ticket_id" not in df.columns:
        df["ticket_id"] = df["fecha"].dt.strftime("%Y-%m-%d") + " " + df["hora"]

    # Derivados
    df["fecha_hora"] = df["fecha"].dt.strftime("%Y-%m-%d") + " " + df["hora"]
    df["dow"] = df["fecha"].dt.dayofweek.map(DOW_MAP)
    if "categoria" not in df.columns:
        df["categoria"] = "Sin categoría"

except Exception as e:
    st.error(f"Error al leer/limpiar el archivo: {e}")
    st.stop()

productos = sorted(df["producto"].astype(str).unique())

# ===== Pestañas =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 KPIs & Alertas", 
    "📈 Gráficos", 
    "🤖 Simulador IA", 
    "💡 Recomendaciones", 
    "📄 PDF"
])

# ===== TAB 1: KPIs & Alertas =====
with tab1:
    st.subheader("📌 Indicadores clave del mes")
    total_ingresos = float((df["cantidad"] * df["precio_unitario"]).sum())
    total_costes = float((df["cantidad"] * df["coste_unitario"]).sum())
    beneficio_total = total_ingresos - total_costes

    tickets = df.groupby("ticket_id")["total"].sum()
    n_tickets = int(len(tickets))
    ticket_medio = total_ingresos / n_tickets if n_tickets > 0 else 0.0

    # Producto top por cantidad
    prod_qty = df.groupby("producto")["cantidad"].sum().sort_values(ascending=False)
    producto_top = prod_qty.index[0] if len(prod_qty) else "N/A"
    cantidad_top = int(prod_qty.iloc[0]) if len(prod_qty) else 0

    # Día fuerte/flojo por cantidad
    dia_qty = df.groupby("dow")["cantidad"].sum().reindex(DOW_ORDER).fillna(0)
    dia_mas_ventas = dia_qty.idxmax() if len(dia_qty) else "—"
    dia_menos_ventas = dia_qty.idxmin() if len(dia_qty) else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingresos totales", eur(total_ingresos))
    c2.metric("Beneficio total", eur(beneficio_total))
    c3.metric("Ticket medio", eur(ticket_medio))
    c4, c5, c6 = st.columns(3)
    c4.metric("Producto más vendido", f"{producto_top} ({cantidad_top} uds)")
    c5.metric("Día más fuerte", dia_mas_ventas)
    c6.metric("Día más flojo", dia_menos_ventas)

    st.subheader("🚨 Alertas automáticas")
    alertas = []
    # Rentabilidad media por producto
    if "precio_unitario" in df.columns:
        df["margen_unit"] = (df["precio_unitario"] - df["coste_unitario"]).fillna(0)
        for p in productos:
            dp = df[df["producto"] == p]
            if len(dp) == 0: 
                continue
            rentabilidad = float(dp["margen_unit"].mean())
            if rentabilidad < 0.10:
                alertas.append(f"❗ Rentabilidad baja en {p}: {rentabilidad:.2f} €/unidad")
    if alertas:
        for a in alertas:
            st.warning(a)
    else:
        st.success("✅ No se detectaron alertas importantes.")

# ===== TAB 2: Gráficos =====
with tab2:
    # ----- Predicción con Prophet -----
    st.subheader("📈 Predicción de ventas con Prophet (14 días)")

    df_prophet = df.groupby("fecha")["cantidad"].sum().reset_index()
    df_prophet = df_prophet.rename(columns={"fecha": "ds", "cantidad": "y"})

    if len(df_prophet) >= 7:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=14)
        forecast = m.predict(future)

        fig_forecast = px.line(forecast, x="ds", y="yhat", title="Predicción de ventas (14 días)")
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Se necesitan al menos 7 días de datos para entrenar la predicción.")

    # ----- Ventas por hora (media) -----
    st.subheader("⏰ Ventas por hora (media)")
    try:
        df["hora_int"] = pd.to_datetime(df["hora"], format="%H:%M", errors="coerce").dt.hour
        df_hora = df.groupby("hora_int")["cantidad"].mean().reset_index().dropna()
        fig_hora = px.bar(df_hora, x="hora_int", y="cantidad", title="Ventas medias por hora")
        st.plotly_chart(fig_hora, use_container_width=True)
    except Exception:
        st.warning("No hay datos de hora válidos.")

    # ----- Ventas por día de la semana -----
    st.subheader("📆 Ventas por día de la semana")
    df_dow = df.groupby("dow")["cantidad"].sum().reindex(DOW_ORDER).reset_index()
    df_dow.columns = ["Día","Cantidad"]
    fig_dow = px.bar(df_dow, x="Día", y="Cantidad", title="Ventas por día de la semana")
    st.plotly_chart(fig_dow, use_container_width=True)

    # === Normalizar columnas para los bloques extra ===
    df_std = df.copy()
    df_std.columns = [c.strip().lower() for c in df_std.columns]

    if "hora" not in df_std.columns:
        df_std["hora"] = "09:00"
    if "ticket_id" not in df_std.columns:
        df_std["ticket_id"] = df_std["fecha"].dt.strftime("%Y-%m-%d") + " " + df_std["hora"]

    if "total" not in df_std.columns or df_std["total"].isna().all():
        if "precio_unitario" in df_std.columns and "cantidad" in df_std.columns:
            df_std["total"] = (pd.to_numeric(df_std["cantidad"], errors="coerce").fillna(0) *
                               pd.to_numeric(df_std["precio_unitario"], errors="coerce").fillna(0))
        else:
            df_std["total"] = 0.0

    # ----- Heatmap ingresos hora x día -----
    st.subheader("🔥 Mapa de calor: ingresos por hora y día")
    tmp = df_std.copy()
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce")
    tmp["hora"] = tmp["hora"].astype(str).str[:5]
    tmp["hora_int"] = pd.to_datetime(tmp["hora"], format="%H:%M", errors="coerce").dt.hour
    tmp["ingresos"] = pd.to_numeric(tmp["total"], errors="coerce").fillna(0)
    dow_map = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    tmp["dow"] = tmp["fecha"].dt.dayofweek.map(dow_map)

    pivot = (tmp.dropna(subset=["hora_int","dow"])
               .pivot_table(index="dow", columns="hora_int", values="ingresos", aggfunc="sum")
               .reindex(DOW_ORDER)
               .fillna(0))
    if not pivot.empty:
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Hora", y="Día", color="€"),
            title="Mapa de calor de ingresos (día × hora)",
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No hay datos suficientes de hora para el mapa de calor.")

    # ----- Pie % ingresos por producto (Top 10) -----
    st.subheader("🥧 % de ingresos por producto (Top 10)")
    if "producto" in df_std.columns:
        ing_por_prod = (df_std.groupby("producto")["total"].sum()
                        .sort_values(ascending=False).head(10))
        if len(ing_por_prod) > 0:
            fig_pie = px.pie(
                values=ing_por_prod.values,
                names=ing_por_prod.index,
                title="Porcentaje de ingresos por producto (Top 10)"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay suficientes productos para el gráfico de porcentajes.")
    else:
        st.warning("No se encontró la columna 'producto'.")

    # ----- Ticket medio por día -----
    st.subheader("🧾 Ticket medio por día (€)")
    tix = (df_std.groupby(["fecha","ticket_id"])["total"].sum()
           .reset_index().rename(columns={"total":"importe"}))
    ticket_medio_diario = (tix.groupby("fecha")["importe"].mean()
                           .reset_index(name="ticket_medio"))
    fig_tm = px.line(ticket_medio_diario, x="fecha", y="ticket_medio", title="Ticket medio diario (€)")
    st.plotly_chart(fig_tm, use_container_width=True)

    # ----- Comparativa última semana vs anterior -----
    st.subheader("📅 Ingresos: última semana vs anterior")
    daily_ing = (df_std.groupby("fecha")["total"].sum()
                 .reset_index().sort_values("fecha"))
    if len(daily_ing) >= 2:
        daily_ing["week_start"] = (daily_ing["fecha"] -
                                   pd.to_timedelta(daily_ing["fecha"].dt.weekday, unit="D"))
        weekly = (daily_ing.groupby("week_start")["total"].sum()
                  .reset_index().sort_values("week_start"))
        if len(weekly) >= 2:
            last2 = weekly.tail(2)
            s_prev, s_ult = last2.iloc[0], last2.iloc[1]
            dif_abs = s_ult["total"] - s_prev["total"]
            dif_pct = (dif_abs / s_prev["total"] * 100) if s_prev["total"] > 0 else np.nan

            cA, cB, cC = st.columns(3)
            cA.metric("Semana anterior", f"{s_prev['total']:.2f} €")
            cB.metric("Última semana", f"{s_ult['total']:.2f} €", f"{dif_abs:+.2f} €")
            cC.metric("Variación %", f"{dif_pct:+.1f}%")

            fig_sem = px.bar(
                last2.assign(semana=last2["week_start"].dt.strftime("%d %b")),
                x="semana", y="total", title="Ingresos por semana (comparativa)"
            )
            st.plotly_chart(fig_sem, use_container_width=True)
        else:
            st.info("Se necesitan al menos 2 semanas distintas en los datos.")
    else:
        st.info("Se necesitan al menos 2 días para comparar semanas.")

# ===== TAB 3: Simulador IA (lineal) =====
with tab3:
    st.subheader("🤖 Simulación IA: Precio óptimo por producto (lineal)")
    for p in productos:
        dp = df[df["producto"] == p]
        if dp["precio_unitario"].nunique() < 2 or len(dp) < 8:
            continue
        X = dp["precio_unitario"].values.reshape(-1,1)
        y = dp["cantidad"].values
        model = LinearRegression()
        model.fit(X, y)

        pmin = float(max(0.01, dp["precio_unitario"].min()*0.7))
        pmax = float(dp["precio_unitario"].max()*1.5)
        precios_sim = np.linspace(pmin, pmax, 60)
        demanda_sim = model.predict(precios_sim.reshape(-1,1))
        demanda_sim = np.clip(demanda_sim, 0, None)  # no negativa

        ingreso = precios_sim * demanda_sim
        precio_opt = float(precios_sim[np.argmax(ingreso)])

        st.markdown(f"**{p}** — Precio óptimo estimado: **{precio_opt:.2f} €**")
        st.slider("Ajusta el precio",
                  min_value=float(pmin),
                  max_value=float(pmax),
                  value=float(precio_opt),
                  step=0.05,
                  key=f"slider_{p}")

# ===== TAB 4: Recomendaciones (combos) =====
with tab4:
    st.subheader("💡 Recomendaciones de combos")
    combinaciones = df.groupby("ticket_id")["producto"].apply(list)
    pares = []
    for lista in combinaciones:
        lista = list(set(map(str, lista)))
        for i in range(len(lista)):
            for j in range(i+1, len(lista)):
                pares.append(tuple(sorted((lista[i], lista[j]))))
    top_combos = Counter(pares).most_common(5)
    if not top_combos:
        st.info("No hay suficientes tickets con ≥2 productos para sugerir combos.")
    else:
        for (p1, p2), count in top_combos:
            v1 = df[df["producto"] == p1]["precio_unitario"].mean()
            v2 = df[df["producto"] == p2]["precio_unitario"].mean()
            if pd.isna(v1) or pd.isna(v2):
                continue
            sugerido = round((v1 + v2) * 0.90, 2)  # -10% pack
            st.markdown(f"🧩 *{p1} + {p2}* apareció en **{count}** tickets. "
                        f"Propón el combo a **{sugerido:.2f} €** (-10%).")

# ===== TAB 5: PDF =====
with tab5:
    st.subheader("📄 Exportar informe PDF")

    def crear_pdf_bytes() -> bytes:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        # Nota: fpdf (no fpdf2) no soporta UTF-8; usamos latin-1 con reemplazo
        def w(text, size=12, bold=False, ln=True, align="L"):
            if bold:
                pdf.set_font("Arial", "B", size)
            else:
                pdf.set_font("Arial", "", size)
            safe = text.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 8 if size<=12 else 10, safe, align=align)
            if not ln:
                pdf.ln(0)

        pdf.set_font("Arial", "B", 16)
        w("Informe de Ventas - Cafetería", size=16, bold=True, align="C")
        pdf.ln(2)

        w(f"Ingresos: {eur(total_ingresos)}")
        w(f"Beneficio: {eur(beneficio_total)}")
        w(f"Ticket medio: {eur(ticket_medio)}")
        w(f"Producto más vendido: {producto_top} ({cantidad_top} uds)")
        w(f"Día más fuerte: {dia_mas_ventas}")
        w(f"Día más flojo: {dia_menos_ventas}")

        pdf.ln(2); w("Alertas:", size=14, bold=True)
        if 'alertas' in locals() and alertas:
            for a in alertas:
                w(f"- {a}")
        else:
            w("No se detectaron alertas.")

        return pdf.output(dest="S").encode("latin-1", "ignore")

    if st.button("📥 Generar PDF"):
        data = crear_pdf_bytes()
        st.download_button("Descargar informe", data, file_name="informe_cafeteria.pdf", mime="application/pdf")