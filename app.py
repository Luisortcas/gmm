from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
K_VALUES = range(2, 9)
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

st.set_page_config(
    page_title="Perfiles de demanda PJM",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_hourly_data(csv_path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path.name}")

    raw = pd.read_csv(path)
    required = {"Datetime", "PJME_MW"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(
            "El CSV debe incluir las columnas Datetime y PJME_MW. "
            f"Faltan: {', '.join(sorted(missing_columns))}."
        )

    cleaned = raw.loc[:, ["Datetime", "PJME_MW"]].copy()
    cleaned["Datetime"] = pd.to_datetime(cleaned["Datetime"], errors="coerce")
    cleaned["PJME_MW"] = pd.to_numeric(cleaned["PJME_MW"], errors="coerce")
    invalid_rows = int(cleaned[["Datetime", "PJME_MW"]].isna().any(axis=1).sum())
    cleaned = cleaned.dropna(subset=["Datetime", "PJME_MW"])
    duplicate_timestamps = int(cleaned["Datetime"].duplicated().sum())

    if cleaned.empty:
        raise ValueError("El archivo no contiene lecturas horarias válidas.")

    hourly = (
        cleaned.groupby("Datetime", as_index=True)["PJME_MW"]
        .mean()
        .sort_index()
        .to_frame()
    )
    hourly.index.name = "Datetime"
    quality = {
        "rows": len(raw),
        "invalid_rows": invalid_rows,
        "duplicate_timestamps": duplicate_timestamps,
    }
    return hourly, quality


@st.cache_data(show_spinner=False)
def make_daily_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    indexed = hourly.assign(
        Fecha=hourly.index.normalize(),
        Hora=hourly.index.hour,
    )
    profiles = indexed.pivot_table(
        index="Fecha",
        columns="Hora",
        values="PJME_MW",
        aggfunc="mean",
    ).reindex(columns=range(24))
    profiles.columns.name = "Hora"
    profiles.index.name = "Fecha"
    return profiles.dropna(axis=0, how="any")


@st.cache_data(show_spinner=False)
def run_analysis(daily_profiles: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
    if len(daily_profiles) < max(K_VALUES) + 1:
        raise ValueError(
            "Se requieren al menos 9 días completos para comparar todos los "
            "valores de K entre 2 y 8."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(daily_profiles)
    metrics = []
    kmeans_models = {}
    gmm_models = {}

    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=5,
        )
        gmm.fit(scaled)

        km_labels = kmeans.labels_
        gmm_labels = gmm.predict(scaled)
        kmeans_models[k] = (kmeans, km_labels)
        gmm_models[k] = (gmm, gmm_labels)
        metrics.append(
            {
                "K": k,
                "Inercia K-Means": kmeans.inertia_,
                "Silueta K-Means": silhouette_score(scaled, km_labels),
                "Silueta GMM": silhouette_score(scaled, gmm_labels),
                "AIC GMM": gmm.aic(scaled),
                "BIC GMM": gmm.bic(scaled),
                "ARI K-Means vs GMM": adjusted_rand_score(km_labels, gmm_labels),
            }
        )

    kmeans, km_labels = kmeans_models[3]
    gmm, gmm_labels = gmm_models[4]
    distances = kmeans.transform(scaled).min(axis=1)
    log_likelihood = gmm.score_samples(scaled)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(12, 4, 12),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    autoencoder.fit(scaled, scaled)
    reconstruction = autoencoder.predict(scaled)
    reconstruction_error = np.mean((scaled - reconstruction) ** 2, axis=1)

    results = pd.DataFrame(
        {
            "Perfil_KMeans": km_labels,
            "Componente_GMM": gmm_labels,
            "Distancia_KMeans": distances,
            "LogProb_GMM": log_likelihood,
            "Error_AE": reconstruction_error,
            "Consumo_Medio_MW": daily_profiles.mean(axis=1).to_numpy(),
            "Pico_Max_MW": daily_profiles.max(axis=1).to_numpy(),
        },
        index=daily_profiles.index,
    )
    results.index.name = "Fecha"
    results["Anomalia_KMeans"] = results["Distancia_KMeans"] > np.percentile(
        distances, 99
    )
    results["Anomalia_GMM"] = results["LogProb_GMM"] < np.percentile(
        log_likelihood, 1
    )
    results["Anomalia_AE"] = results["Error_AE"] > np.percentile(
        reconstruction_error, 99
    )
    results["Consenso_Anomalia"] = results[
        ["Anomalia_KMeans", "Anomalia_GMM", "Anomalia_AE"]
    ].sum(axis=1)

    average_demand = results.groupby("Perfil_KMeans")["Consumo_Medio_MW"].mean()
    ordered_labels = average_demand.sort_values().index.tolist()
    descriptors = ["menor demanda", "demanda intermedia", "mayor demanda"]
    profile_names = {
        label: f"Perfil {chr(ord('A') + position)} · "
        f"{descriptors[position] if len(ordered_labels) == 3 else 'grupo ' + str(position + 1)}"
        for position, label in enumerate(ordered_labels)
    }
    results["Perfil"] = results["Perfil_KMeans"].map(profile_names)

    summary = (
        results.groupby("Perfil", sort=False)
        .agg(
            Dias=("Consumo_Medio_MW", "size"),
            Porcentaje=("Consumo_Medio_MW", lambda values: 100 * len(values) / len(results)),
            Demanda_Media_MW=("Consumo_Medio_MW", "mean"),
            Pico_Medio_MW=("Pico_Max_MW", "mean"),
            Anomalias_AE=("Anomalia_AE", "sum"),
        )
        .reset_index()
    )
    curves = (
        daily_profiles.assign(Perfil=results["Perfil"])
        .groupby("Perfil", sort=False)[list(range(24))]
        .mean()
    )
    month_share = pd.crosstab(
        results["Perfil"],
        results.index.month,
        normalize="index",
    ).reindex(index=summary["Perfil"], columns=range(1, 13), fill_value=0) * 100

    metrics_df = pd.DataFrame(metrics)
    return {
        "metrics": metrics_df,
        "results": results,
        "summary": summary,
        "curves": curves,
        "month_share": month_share,
        "autoencoder_iterations": int(autoencoder.n_iter_),
    }


def main() -> None:
    st.title("Perfiles y anomalías de la demanda eléctrica PJM")
    st.caption("Análisis no supervisado de perfiles diarios · Autores: Luis Ortega y Nicolas Carvajal")

    available_files = [
        ROOT / name
        for name in ("PJME_hourly.csv", "data.csv")
        if (ROOT / name).is_file()
    ]
    if not available_files:
        st.error("No se encontró PJME_hourly.csv ni data.csv junto a app.py.")
        st.stop()

    file_names = [path.name for path in available_files]
    default_index = file_names.index("PJME_hourly.csv") if "PJME_hourly.csv" in file_names else 0
    selected_file = st.sidebar.selectbox(
        "Fuente de datos",
        file_names,
        index=default_index,
    )

    try:
        hourly, quality = load_hourly_data(str(ROOT / selected_file))
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    years = sorted(hourly.index.year.unique().tolist())
    selected_years = st.sidebar.multiselect("Años incluidos", years, default=years)
    if not selected_years:
        st.warning("Selecciona al menos un año para continuar.")
        st.stop()

    filtered_hourly = hourly.loc[hourly.index.year.isin(selected_years)]
    daily = make_daily_profiles(filtered_hourly)
    incomplete_days = int(filtered_hourly.index.normalize().nunique() - len(daily))

    st.markdown(
        """
        **Problema.** Identificar patrones diarios recurrentes en la demanda eléctrica
        y señalar días atípicos sin etiquetas previas. Los perfiles pueden apoyar la
        exploración y planificación; no reemplazan un pronóstico operativo.
        """
    )
    st.markdown(
        """
        **Datos.** Serie horaria de demanda PJM en MW. El repositorio incluye
        `PJME_hourly.csv` y una versión con variables de calendario (`data.csv`);
        ambas se preparan usando las lecturas `Datetime` y `PJME_MW`.
        """
    )

    first_col, second_col, third_col, fourth_col = st.columns(4)
    first_col.metric("Lecturas horarias", f"{quality['rows']:,}")
    second_col.metric(
        "Periodo seleccionado",
        f"{filtered_hourly.index.min():%Y-%m-%d} – {filtered_hourly.index.max():%Y-%m-%d}",
    )
    third_col.metric("Días completos", f"{len(daily):,}")
    fourth_col.metric("Marcas duplicadas agregadas", f"{quality['duplicate_timestamps']:,}")

    with st.expander("Preprocesamiento y calidad de datos"):
        st.write(
            f"Se descartaron {quality['invalid_rows']:,} filas con fecha o demanda "
            f"no válida y se promediaron {quality['duplicate_timestamps']:,} marcas "
            "de tiempo duplicadas. La demanda se reorganiza como una matriz de "
            "fecha × 24 horas; se excluyen días que no contienen las 24 horas. "
            "No se imputan horas faltantes."
        )
        st.write(
            f"En el periodo seleccionado se excluyeron {incomplete_days:,} fechas "
            "incompletas. Cada hora se estandariza con `StandardScaler` antes del "
            "ajuste, para que las escalas horarias sean comparables."
        )

    st.markdown(
        """
        **Metodología.** Se comparan K-Means y mezclas gaussianas (GMM) para
        `K = 2…8`. K-Means usa `n_init=10`; GMM usa covarianza completa,
        `n_init=5` y EM. Se reportan silueta, inercia, AIC, BIC y ARI entre
        particiones. El modelo final de perfiles usa K-Means (`K=3`) y el
        componente probabilístico usa GMM (`K=4`). Para anomalías se combinan
        distancia a centroide, log-verosimilitud GMM y error de reconstrucción de
        un autoencoder MLP `24→12→4→12→24`; los umbrales son los percentiles 99,
        1 y 99, respectivamente.
        """
    )

    analysis_key = (selected_file, tuple(selected_years))
    if st.button("Ejecutar análisis completo", type="primary"):
        st.session_state["analysis_key"] = analysis_key

    if st.session_state.get("analysis_key") != analysis_key:
        st.info(
            "Ejecuta el análisis para calcular las métricas, perfiles, anomalías "
            "y visualizaciones interactivas."
        )
        monthly = filtered_hourly["PJME_MW"].resample("MS").mean().rename("Demanda media (MW)").reset_index()
        overview = px.line(
            monthly,
            x="Datetime",
            y="Demanda media (MW)",
            title="Demanda media mensual",
            labels={"Datetime": "Fecha"},
        )
        overview.update_layout(hovermode="x unified")
        st.plotly_chart(overview, width="stretch")
        st.stop()

    try:
        with st.spinner("Ajustando modelos y calculando métricas. Esto puede tardar un momento…"):
            analysis = run_analysis(daily)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    metrics = analysis["metrics"]
    results = analysis["results"]
    summary = analysis["summary"]

    overview_tab, methods_tab, profiles_tab, anomalies_tab, conclusions_tab = st.tabs(
        ["Resumen", "Metodología y comparación", "Perfiles", "Anomalías", "Conclusiones"]
    )

    with overview_tab:
        best_silhouette = metrics.loc[metrics["Silueta K-Means"].idxmax()]
        best_bic = metrics.loc[metrics["BIC GMM"].idxmin()]
        st.subheader("Hallazgos principales")
        st.write(
            f"En el barrido actual, la mayor silueta de K-Means se obtiene con "
            f"K={int(best_silhouette['K'])} ({best_silhouette['Silueta K-Means']:.3f}). "
            f"El BIC mínimo de GMM aparece en K={int(best_bic['K'])}; como el mínimo "
            "está en el límite del rango evaluado, no demuestra que ese sea el número "
            "óptimo global de componentes."
        )
        st.write(
            "La silueta mide separación interna, no exactitud predictiva. No existen "
            "etiquetas de referencia para validar los grupos como clases verdaderas."
        )

        summary_display = summary.copy()
        summary_display[["Porcentaje", "Demanda_Media_MW", "Pico_Medio_MW"]] = (
            summary_display[["Porcentaje", "Demanda_Media_MW", "Pico_Medio_MW"]].round(1)
        )
        st.dataframe(summary_display, hide_index=True, width="stretch")

        monthly = filtered_hourly["PJME_MW"].resample("MS").mean().rename("Demanda media (MW)").reset_index()
        monthly_fig = px.line(
            monthly,
            x="Datetime",
            y="Demanda media (MW)",
            title="Demanda media mensual en el periodo seleccionado",
            labels={"Datetime": "Fecha"},
        )
        monthly_fig.update_layout(hovermode="x unified")
        st.plotly_chart(monthly_fig, width="stretch")

    with methods_tab:
        st.subheader("Comparación de modelos")
        left, right = st.columns(2)
        silhouette_fig = go.Figure()
        silhouette_fig.add_trace(
            go.Scatter(
                x=metrics["K"],
                y=metrics["Silueta K-Means"],
                name="K-Means",
                mode="lines+markers",
            )
        )
        silhouette_fig.add_trace(
            go.Scatter(
                x=metrics["K"],
                y=metrics["Silueta GMM"],
                name="GMM",
                mode="lines+markers",
            )
        )
        silhouette_fig.update_layout(
            title="Silueta por número de grupos",
            xaxis_title="K",
            yaxis_title="Silueta (mayor es mejor)",
            hovermode="x unified",
        )
        left.plotly_chart(silhouette_fig, width="stretch")

        bic_fig = px.line(
            metrics,
            x="K",
            y="BIC GMM",
            markers=True,
            title="BIC de GMM (menor es mejor)",
            labels={"K": "Componentes", "BIC GMM": "BIC"},
        )
        right.plotly_chart(bic_fig, width="stretch")

        ari_fig = px.line(
            metrics,
            x="K",
            y="ARI K-Means vs GMM",
            markers=True,
            title="Acuerdo entre particiones (ARI)",
            labels={"K": "K", "ARI K-Means vs GMM": "ARI"},
        )
        st.plotly_chart(ari_fig, width="stretch")

        display_metrics = metrics.round(4)
        st.dataframe(display_metrics, hide_index=True, width="stretch")
        st.download_button(
            "Descargar métricas CSV",
            data=metrics.to_csv(index=False).encode("utf-8-sig"),
            file_name="metricas_modelos_pjm.csv",
            mime="text/csv",
        )
        st.markdown(
            """
            **Lectura.** Una silueta mayor indica grupos más separados en la
            representación estandarizada; el BIC penaliza la complejidad del GMM.
            La inercia disminuye al aumentar K por construcción. El ARI compara
            cuánto coinciden dos particiones, no cuál es correcta.
            """
        )

    with profiles_tab:
        st.subheader("Perfiles diarios K-Means (K=3)")
        profile_fig = go.Figure()
        for profile, row in analysis["curves"].iterrows():
            profile_fig.add_trace(
                go.Scatter(
                    x=list(range(24)),
                    y=row.to_numpy(),
                    mode="lines",
                    name=profile,
                    hovertemplate="Hora %{x}:00<br>Demanda %{y:,.0f} MW<extra></extra>",
                )
            )
        profile_fig.update_layout(
            title="Curva horaria media por perfil",
            xaxis_title="Hora del día",
            yaxis_title="Demanda media (MW)",
            xaxis=dict(dtick=2),
            hovermode="x unified",
        )
        st.plotly_chart(profile_fig, width="stretch")

        month_share = analysis["month_share"].copy()
        month_share.columns = MONTHS
        heatmap = px.imshow(
            month_share,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels={"x": "Mes", "y": "Perfil", "color": "% de días"},
            title="Distribución de los días de cada perfil por mes",
        )
        heatmap.update_traces(texttemplate="%{z:.0f}%")
        st.plotly_chart(heatmap, width="stretch")

        st.caption(
            "Los nombres describen la demanda media relativa de cada grupo; la "
            "asociación estacional debe verificarse en el mapa mensual."
        )

    with anomalies_tab:
        st.subheader("Días atípicos")
        anomalous = results.loc[results["Anomalia_AE"]].copy()
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=results.index,
                y=results["Error_AE"],
                mode="lines",
                name="Error de reconstrucción",
                line={"color": "#4C78A8", "width": 1},
            )
        )
        chart.add_trace(
            go.Scatter(
                x=anomalous.index,
                y=anomalous["Error_AE"],
                mode="markers",
                name="Sobre el P99 del autoencoder",
                marker={"color": "#D62728", "size": 6},
                customdata=anomalous[["Consumo_Medio_MW", "Consenso_Anomalia"]].to_numpy(),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Error: %{y:.4f}"
                    "<br>Media: %{customdata[0]:,.0f} MW"
                    "<br>Modelos en consenso: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
        threshold = float(results["Error_AE"].quantile(0.99))
        chart.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Umbral P99: {threshold:.3f}",
        )
        chart.update_layout(
            title="Error de reconstrucción por día",
            xaxis_title="Fecha",
            yaxis_title="Error cuadrático medio",
            hovermode="x unified",
        )
        st.plotly_chart(chart, width="stretch")

        st.write(
            f"El autoencoder marca {len(anomalous):,} días sobre el percentil 99 "
            f"({100 * len(anomalous) / len(results):.2f}% en esta muestra). "
            "Un día atípico es una señal para investigar, no evidencia por sí sola "
            "de un error o evento real."
        )
        top_anomalies = (
            results.sort_values("Error_AE", ascending=False)
            .head(10)
            .reset_index()
            .loc[
                :,
                [
                    "Fecha",
                    "Perfil",
                    "Consumo_Medio_MW",
                    "Pico_Max_MW",
                    "Error_AE",
                    "LogProb_GMM",
                    "Consenso_Anomalia",
                ],
            ]
        )
        st.dataframe(
            top_anomalies.round(
                {
                    "Consumo_Medio_MW": 1,
                    "Pico_Max_MW": 1,
                    "Error_AE": 4,
                    "LogProb_GMM": 2,
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Descargar resultados diarios CSV",
            data=results.reset_index().to_csv(index=False).encode("utf-8-sig"),
            file_name="resultados_diarios_pjm.csv",
            mime="text/csv",
        )

    with conclusions_tab:
        st.subheader("Conclusiones")
        st.markdown(
            """
            - K-Means ofrece una segmentación interpretable de las curvas horarias;
              la silueta ayuda a comparar K, pero no determina por sí sola la
              utilidad operativa de un perfil.
            - GMM representa grupos con distribuciones y covarianzas más flexibles.
              Un BIC que sigue mejorando en el máximo K probado indica que conviene
              ampliar el barrido antes de afirmar un óptimo.
            - Las tres señales de anomalía aportan perspectivas distintas. Los
              días detectados requieren contraste con eventos, clima y calidad
              de medición antes de tomar decisiones.
            """
        )
        st.markdown("**Limitaciones y sesgos identificados**")
        st.markdown(
            """
            - Los datos contienen demanda agregada de PJM, no observaciones de
              generación, precios, clima, festivos ni causas de eventos; por tanto,
              las explicaciones estacionales son descriptivas y no causales.
            - La serie termina en 2018 y no representa necesariamente las
              condiciones actuales. Los cambios de horario y días incompletos
              pueden excluir observaciones de forma no aleatoria.
            - La evaluación es no supervisada y se realiza sobre los mismos días
              usados para ajustar los modelos; no mide desempeño fuera de muestra.
              Los umbrales percentilares fuerzan una fracción extrema por diseño.
            - ARI compara K-Means y GMM, no una verdad de terreno. Diferencias en
              sus grupos no permiten declarar que uno sea más preciso.
            """
        )
        st.markdown("**Trabajo futuro**")
        st.markdown(
            """
            Incorporar clima, calendario de festivos y variables operativas;
            validar con particiones temporales; contrastar anomalías con registros
            de eventos; probar ventanas móviles y otros métodos robustos; y ampliar
            el rango de K y el periodo con datos más recientes.
            """
        )
        st.caption(
            f"El autoencoder terminó en {analysis['autoencoder_iterations']} "
            "iteraciones. El error se calcula sobre los mismos perfiles usados "
            "para el ajuste; no debe interpretarse como una evaluación de "
            "generalización."
        )


if __name__ == "__main__":
    main()
