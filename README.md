# ML_LAI_Estimation

El proyecto aborda un problema de **regresión supervisada** para predecir el **Índice de Área Foliar (LAI)** de la vegetación utilizando datos de reflectancia espectral del satélite **Sentinel-2**. El objetivo es construir un modelo de machine learning que relacione las bandas espectrales con mediciones in situ de LAI, obtenidas de la red **NEON**, para estimar el LAI a gran escala espacial y temporal.

## Dataset de Satelite:

El dataset de satélite contiene datos de reflectancia espectral obtenidos por el satélite Sentinel-2. Los datos son descargados de la plataforma Google Earth Engine con el script `collect_dataset_sat.ipynb` en Google Colab <a target="_blank" href="https://colab.research.google.com/github/esjoal/S2_ML_LAI_Estimation/blob/main/src/result_notebooks/collect_dataset_sat.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>  y grabado en 'src/data/'.

**Columnas del dataset:**
1. **Site_sat**: Nombre del sitio donde se tomaron las mediciones satelitales.
2. **time**: Fecha y hora de adquisición de la imagen satelital.
3. **QA60**: Indicador de calidad de los píxeles. Se utiliza para identificar píxeles con nubes.
4. **SCL**: Clasificación de la superficie según Sentinel-2. Se utiliza para identificar píxeles de vegetación.
5. **B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12**: Bandas espectrales de Sentinel-2. Estas bandas representan diferentes longitudes de onda, desde el visible (azul, verde, rojo) hasta el infrarrojo cercano y de onda corta.
6. **longitude**: Longitud geográfica del píxel.
7. **latitude**: Latitud geográfica del píxel.

Este dataset es la base para entrenar modelos de machine learning que predicen el LAI a partir de las bandas espectrales.

## Dataset insitu: 

El dataset in situ contiene mediciones del Índice de Área Foliar (LAI) obtenidas en campo por la red **NEON** (National Ecological Observatory Network). Los valores de LAI se derivan de fotografías digitales hemisféricas (DHP) que fueron procesadas por el Proyecto GBOV del Programa Copernicus para obtener el LAI. El dataset fue descargado de la web: https://gbov.land.copernicus.eu/data-access como una colección de archivos (ver '\data\COPERNICUS_GBOV_RM7_20253103525'). Luego el dataset es preprocesado usando `collect_dataset_insitu.ipynb` y grabado en 'src/data/'.  

**Columnas del dataset:**

1. **Site**: Nombre del sitio donde se realizaron las mediciones in situ.
2. **TIME_IS**: Fecha y hora de la medición in situ.
3. **LAI_Warren_up**: Valor del LAI medido en la dirección ascendente utilizando el método Warren.
4. **LAI_Warren_down**: Valor del LAI medido en la dirección descendente utilizando el método Warren.
5. **LAI_Warren**: Valor combinado del LAI (suma de las mediciones ascendentes y descendentes) calculado para el método Warren.
6. **LAI_Warren_err**: Error asociado a la medición del LAI utilizando el método Warren.
7. **clumping_Warren_up**: Factor de agrupamiento medido en la dirección ascendente para el método Warren.
8. **clumping_Warren_down**: Factor de agrupamiento medido en la dirección descendente para el método Warren.
9. **clumping_Warren_up_err**: Error asociado al factor de agrupamiento ascendente.
10. **clumping_Warren_down_err**: Error asociado al factor de agrupamiento descendente.
11. **IGBP_class**: Clase de cobertura terrestre según la clasificación del **International Geosphere-Biosphere Programme (IGBP)** (e.g., bosques, pastizales, etc.).
12. **latitude**: Latitud geográfica del sitio de medición.
13. **longitude**: Longitud geográfica del sitio de medición.
14. **up_flag**: Indicador de calidad para las mediciones ascendentes (0 indica buena calidad).
15. **down_flag**: Indicador de calidad para las mediciones descendentes (0 indica buena calidad).

Este dataset es esencial para proporcionar las etiquetas (valores reales de LAI) necesarias para entrenar y evaluar los modelos de machine learning.



## Pasos principales del proyecto:

1. **Preparación de datos**

2. **Combinación de datasets**

3. **Exploración de datos (EDA)**

4. **Split de datasets**

5. **Selección de features**

6. **Entrenamiento de modelos**

7. **Modelo final**

8. **Análisis de errores**

9. **Resultados visuales**

## Conclusión:
El modelo de regresión lineal es capaz de predecir el LAI con un desempeño aceptable, siendo más preciso en valores intermedios de LAI. Aunque otros modelos como SVR mostraron ligeras mejoras, se optó por la regresión lineal por su simplicidad y resultados consistentes.

## Características principales

- **Procesamiento de datos satelitales**: Extracción y preprocesamiento de bandas espectrales de Sentinel-2.
- **Modelos de aprendizaje automático**: Implementación de algoritmos como regresión lineal, SVR y árboles de decisión.
- **Evaluación del modelo**: Métricas de rendimiento como RMSE y R² para validar la precisión del modelo.

## Requisitos

- Python 3.8 o superior
- Bibliotecas: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/josea/ML_LAI_Estimation.git
    ```
2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Recolectamos los datos de satélite corriendo el script `collect_dataset_sat.ipynb` en Google Colab (opcional).
2. Recolectamos los datos insitu utilizando el script `collect_dataset_insitu.ipynb`(opcional).
3. Preprocesamos los datos y construimos el modelo paso a paso con `project_ML.ipynb`
4. Entrenamos el modelo ejecutando `pipelines_training.ipynb`.
5. Evaluamos el modelo con `pipelines_testing.ipynb`.

**Nota:** Los pasos 1 y 2  son opcionales ya que los datasets se han incluido previamente en el repositorio.

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request para sugerencias o mejoras.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.