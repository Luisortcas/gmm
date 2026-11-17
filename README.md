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

El dashboard carga los resultados precomputados de `precomputed/` y la
visualización `PJME_Hrs,Mes,Demanda.html`; no ajusta modelos cuando un usuario
abre la app. Mantén esos archivos, `PJME_hourly.csv` y `app.py` en el mismo
repositorio al desplegar.

## Regenerar resultados (opcional)

Solo si cambian los datos o la metodología, ejecuta una vez:

```bash
python -m pip install -r requirements-analysis.txt
python generate_analysis.py
```

El entrenamiento se realiza en esta etapa offline. El uso normal del dashboard
solo lee los CSV y JSON generados.

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

## Publicar y obtener un enlace

Despliega `app.py` en Streamlit Community Cloud, seleccionando este repositorio
y la rama `main`. Asegúrate de subir también la carpeta `precomputed/`, el CSV
de demanda y el archivo HTML; Streamlit instalará las dependencias desde
`requirements.txt`. El enlace de la app se genera al completar el despliegue.
