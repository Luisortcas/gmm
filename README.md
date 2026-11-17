# Perfiles de demanda eléctrica PJM

Aplicación Streamlit para explorar perfiles horarios de demanda, comparar
K-Means y mezclas gaussianas (GMM), y señalar días atípicos mediante un
autoencoder MLP.

**Autores:** Luis Ortega y Nicolas Carvajal

## Ejecutar

Desde la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

La aplicación detecta `PJME_hourly.csv` o `data.csv` en la misma carpeta.
Ambos deben incluir las columnas `Datetime` y `PJME_MW`. El archivo
`PJME_hourly.csv` se selecciona por defecto.

## Metodología

- Se promedian marcas de tiempo duplicadas y se descartan filas sin fecha o
  demanda válida.
- Se forma un perfil por día con 24 valores horarios; los días incompletos se
  excluyen sin imputación.
- Se estandarizan las 24 variables horarias con `StandardScaler`.
- Se comparan K-Means y GMM para K de 2 a 8 usando semilla 42. K-Means usa
  `n_init=10`; GMM usa covarianza completa y `n_init=5`.
- La comparación incluye silueta, inercia, AIC, BIC y ARI entre las
  particiones. No hay etiquetas de referencia, por lo que las métricas no son
  exactitud predictiva.
- El análisis de perfiles usa K-Means con K=3 y el análisis probabilístico
  usa GMM con K=4. El autoencoder es un MLP `24→12→4→12→24`, ReLU, Adam y
  hasta 500 iteraciones.
- Las anomalías se identifican con percentiles extremos de la distancia a
  centroide, log-verosimilitud GMM y error de reconstrucción. Los resultados
  son señales exploratorias que requieren validación externa.

## Alcance y limitaciones

El dataset contiene demanda horaria PJM en MW entre 2002 y 2018. El análisis
es descriptivo, no causal ni un pronóstico operativo. No se incluyen clima,
festivos, precios, generación o registros de eventos. Los modelos se ajustan
y evalúan sobre el mismo periodo; se recomienda validar con particiones
temporales y datos más recientes antes de usar los resultados para decisiones.
