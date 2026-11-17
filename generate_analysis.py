import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "PJME_hourly.csv"
OUTPUT = ROOT / "precomputed"
K_VALUES = range(2, 9)


def main() -> None:
    raw = pd.read_csv(SOURCE, usecols=["Datetime", "PJME_MW"])
    raw_rows = len(raw)
    raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    raw["PJME_MW"] = pd.to_numeric(raw["PJME_MW"], errors="coerce")
    invalid_rows = int(raw[["Datetime", "PJME_MW"]].isna().any(axis=1).sum())
    valid = raw.dropna(subset=["Datetime", "PJME_MW"])
    duplicate_timestamps = int(valid["Datetime"].duplicated().sum())
    hourly = (
        valid.groupby("Datetime")["PJME_MW"]
        .mean()
        .sort_index()
        .to_frame()
    )

    profiled = hourly.assign(
        Fecha=hourly.index.normalize(),
        Hora=hourly.index.hour,
    )
    daily = (
        profiled.pivot_table(
            index="Fecha",
            columns="Hora",
            values="PJME_MW",
            aggfunc="mean",
        )
        .reindex(columns=range(24))
        .dropna(axis=0, how="any")
    )
    if len(daily) < max(K_VALUES) + 1:
        raise ValueError("Se necesitan al menos 9 días completos para calcular K=2…8.")

    scaled = StandardScaler().fit_transform(daily)
    metrics = []
    kmeans_models = {}
    gmm_models = {}
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=5,
        ).fit(scaled)
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
    reconstruction_error = np.mean((scaled - autoencoder.predict(scaled)) ** 2, axis=1)

    results = pd.DataFrame(
        {
            "Perfil_KMeans": km_labels,
            "Componente_GMM": gmm_labels,
            "Distancia_KMeans": distances,
            "LogProb_GMM": log_likelihood,
            "Error_AE": reconstruction_error,
            "Consumo_Medio_MW": daily.mean(axis=1).to_numpy(),
            "Pico_Max_MW": daily.max(axis=1).to_numpy(),
        },
        index=daily.index,
    )
    results.index.name = "Fecha"
    results["Anomalia_KMeans"] = results["Distancia_KMeans"] > np.percentile(distances, 99)
    results["Anomalia_GMM"] = results["LogProb_GMM"] < np.percentile(log_likelihood, 1)
    results["Anomalia_AE"] = results["Error_AE"] > np.percentile(reconstruction_error, 99)
    results["Consenso_Anomalia"] = results[
        ["Anomalia_KMeans", "Anomalia_GMM", "Anomalia_AE"]
    ].sum(axis=1)

    average_demand = results.groupby("Perfil_KMeans")["Consumo_Medio_MW"].mean()
    ordered_labels = average_demand.sort_values().index.tolist()
    descriptors = ["menor demanda", "demanda intermedia", "mayor demanda"]
    profile_names = {
        label: f"Perfil {chr(ord('A') + position)} · {descriptors[position]}"
        for position, label in enumerate(ordered_labels)
    }
    results["Perfil"] = results["Perfil_KMeans"].map(profile_names)

    daily_export = daily.copy()
    daily_export.columns = [f"Hora_{hour:02d}" for hour in daily_export.columns]
    export = results.join(daily_export).reset_index()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUTPUT / "daily_analysis.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT / "model_metrics.csv", index=False)
    metadata = {
        "source": SOURCE.name,
        "raw_rows": raw_rows,
        "invalid_rows": invalid_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "complete_days": len(daily),
        "incomplete_days": int(hourly.index.normalize().nunique() - len(daily)),
        "start_date": daily.index.min().strftime("%Y-%m-%d"),
        "end_date": daily.index.max().strftime("%Y-%m-%d"),
        "autoencoder_iterations": int(autoencoder.n_iter_),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Resultados guardados en {OUTPUT.relative_to(ROOT)} ({len(daily)} días completos).")


if __name__ == "__main__":
    main()
