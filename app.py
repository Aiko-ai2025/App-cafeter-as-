productos = df["Producto"].unique()
categorias = df["Categoria"].unique() if "Categoria" in df.columns else []

# Preparar pestañas
tabs = st.tabs(["📌 KPIs", "📈 Gráficos", "🤖 Simulador IA", "💡 Recomendaciones", "📄 PDF"])

with tabs[0]:
    st.subheader("📌 Indicadores clave")
    total_ingresos = (df["Cantidad"] * df["Precio_unitario"]).sum()
    total_costes = (df["Cantidad"] * df["Coste_unitario"]).sum()
    beneficio_total = total_ingresos - total_costes
    ticket_medio = total_ingresos / df["Fecha_Hora"].nunique()

    df["Dia"] = df["Fecha"].dt.strftime("%A")
    dia_mas_ventas = df.groupby("Dia")["Cantidad"].sum().idxmax()
    dia_menos_ventas = df.groupby("Dia")["Cantidad"].sum().idxmin()

    producto_top = df.groupby("Producto")["Cantidad"].sum().idxmax()
    cantidad_top = df.groupby("Producto")["Cantidad"].sum().max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Ingresos totales", f"{total_ingresos:.2f} €")
    col2.metric("Beneficio total", f"{beneficio_total:.2f} €")
    col3.metric("Ticket medio", f"{ticket_medio:.2f} €")

    col4, col5, col6 = st.columns(3)
    col4.metric("Producto más vendido", f"{producto_top} ({cantidad_top} uds)")
    col5.metric("Día más fuerte", dia_mas_ventas)
    col6.metric("Día más flojo", dia_menos_ventas)

    st.subheader("🚨 Alertas automáticas")
    alertas = []
    for producto in productos:
        df_p = df[df["Producto"] == producto]
        rentabilidad = (df_p["Precio_unitario"] - df_p["Coste_unitario"]).mean()
        if rentabilidad < 0.1:
            alertas.append(f"❗ Rentabilidad muy baja en {producto}: {rentabilidad:.2f} €/unidad")
    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success("✅ No se detectaron alertas importantes.")

with tabs[1]:
    st.subheader("📈 Análisis gráfico")
    df_dia = df.groupby("Fecha")["Cantidad"].sum().reset_index()
    df_pred = df_dia.tail(7).copy()
    df_pred["Fecha"] = df_pred["Fecha"] + pd.Timedelta(days=7)
    df_pred["Cantidad"] = df_pred["Cantidad"] * 1.05
    fig_pred = px.line(pd.concat([df_dia, df_pred]), x="Fecha", y="Cantidad", title="📈 Predicción de ventas")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("📊 Ventas por hora")
    df["Hora_int"] = df["Hora"].str[:2].astype(int)
    df_hora = df.groupby("Hora_int")["Cantidad"].mean().reset_index()
    fig_hora = px.bar(df_hora, x="Hora_int", y="Cantidad", title="Ventas medias por hora")
    st.plotly_chart(fig_hora, use_container_width=True)

    st.subheader("📆 Ventas por día")
    df_dia_semana = df.groupby("Dia")["Cantidad"].sum().reset_index()
    fig_dia = px.bar(df_dia_semana, x="Dia", y="Cantidad", title="Ventas por día de la semana")
    st.plotly_chart(fig_dia, use_container_width=True)

with tabs[2]:
    st.subheader("🤖 Simulación IA de precios óptimos")
    for producto in productos:
        df_p = df[df["Producto"] == producto]
        if len(df_p) < 5:
            continue
        precios = df_p["Precio_unitario"]
        cantidades = df_p["Cantidad"]
        X = precios.values.reshape(-1, 1)
        y = cantidades.values
        model = LinearRegression()
        model.fit(X, y)
        precios_simulados = np.linspace(precios.min(), precios.max() * 1.5, 50)
        demanda_simulada = model.predict(precios_simulados.reshape(-1, 1))
        ingreso_simulado = precios_simulados * demanda_simulada
        precio_optimo = precios_simulados[np.argmax(ingreso_simulado)]

        st.markdown(f"**{producto}** — Precio óptimo estimado: **{precio_optimo:.2f} €**")
        st.slider("Ajusta el precio", min_value=float(precios.min()*0.5), max_value=float(precios.max()*1.5), value=float(precio_optimo), step=0.05, key=producto)

with tabs[3]:
    st.subheader("💡 Recomendaciones inteligentes")
    combinaciones = df.groupby("Fecha_Hora")["Producto"].apply(list)
    pares = []
    for lista in combinaciones:
        lista = list(set(lista))
        for i in range(len(lista)):
            for j in range(i+1, len(lista)):
                pares.append(tuple(sorted((lista[i], lista[j]))))
    top_combos = Counter(pares).most_common(5)
    for (prod1, prod2), count in top_combos:
        df1 = df[df["Producto"] == prod1]
        df2 = df[df["Producto"] == prod2]
        sugerido = round((df1["Precio_unitario"].mean() + df2["Precio_unitario"].mean()) * 0.9, 2)
        st.markdown(f"🧩 *{prod1} + {prod2}* se pidió {count} veces. Combo sugerido: **{sugerido:.2f} €**")

with tabs[4]:
    st.subheader("📄 Generar informe PDF")
    def crear_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Informe de Ventas - Cafetería", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, "Resumen general:", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Ingresos: {total_ingresos:.2f} €\nBeneficio: {beneficio_total:.2f} €\nProducto más vendido: {producto_top}")
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Alertas:", ln=True)
        pdf.set_font("Arial", size=11)
        if alertas:
            for a in alertas:
                pdf.multi_cell(0, 8, a)
        else:
            pdf.multi_cell(0, 8, "No se detectaron alertas.")
        pdf.output("informe_cafeteria.pdf")

    if st.button("📥 Descargar informe PDF"):
        crear_pdf()
        with open("informe_cafeteria.pdf", "rb") as f:
            st.download_button("Descargar informe", f, file_name="informe_cafeteria.pdf")
