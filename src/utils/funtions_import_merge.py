import numpy as np
import pandas as pd
import datetime as dt

import os

from variables import COORDS as coords


def satellite_import(file_name_sat):

    # Definimos ruta de los archivos.
    datadir_path = os.path.join('..','data')
    dataset_path = os.path.join(datadir_path, file_name_sat)

    # Importamos el dataset con Pandas
    df_temp = pd.read_csv(dataset_path, sep=',')

    # Eliminamos columna innecesaria de indices importado
    if 'Unnamed: 0' in df_temp.columns: # Comprobamos si 'Unnamed: 0' está en las columnas
        # Eliminar la columna 'Unnamed: 0'
        df_temp.drop(columns=['Unnamed: 0'], inplace=True) 
    
    # Convertimos a tipo datetime la columna 'time' que está en formato Unix epoch time.
    df_temp['time'] = pd.to_datetime(df_temp['time'], unit='ms')

    # Ordenamos el dataframe por Site y time, y reseteamos el indice.
    df_temp.sort_values(by=['Site_sat','time'], inplace=True)
    df_temp.reset_index(drop=True, inplace=True)
    
    # Eliminamos las filas (espectros) correspondientes a pixeles con nubes
    df_temp = df_temp[df_temp['QA60'] == 0] # Me quedo con las filas (pixeles) libre de nubes 

    # Elimino filas de no-vegetación
    df_temp = df_temp[df_temp['SCL'] == 4] # Me quedo con las filas (pixeles) de vegetación

    # Escalamos las bandas a los valores de reflectancia reales (originalmente están multiplicados por 10000)
    bandas = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'] # Definimos una lista con las columnas correspondientes a las bandas
    for banda in bandas: # Recorremos todas las bandas
        df_temp.loc[:,banda] = df_temp[banda].astype(float) # Forzamos el tipo de los datos para hacerlos float.
        df_temp.loc[:,banda] = df_temp[banda] * 0.0001 # Escalamos

    return df_temp


def insitu_import(file_name_insitu):
    # Definimos ruta de los archivos.
    datadir_path = os.path.join('..','data')
    dataset_path = os.path.join(datadir_path, file_name_insitu)


    # Importamos el dataset con Pandas
    df_temp = pd.read_csv(dataset_path, sep=',')


    # Eliminamos columna innecesaria de indices importado
    if 'Unnamed: 0' in df_temp.columns: # Comprobamos si 'Unnamed: 0' está en las columnas
        # Eliminar la columna 'Unnamed: 0'
        df_temp.drop(columns=['Unnamed: 0'], inplace=True) 


    # Convertimos a tipo datetime la columna 'TIME_IS'
    df_temp['TIME_IS'] = pd.to_datetime(df_temp['TIME_IS'])


    # Pasamos todos los valores nulos a NaN
    valores_nulos = [-999.0, 999.0, -999, 999, '-999.0', '999.0', '-999', '999'] # Lista de valores a reemplazar por NaN
    df_temp.replace(valores_nulos, np.nan, inplace=True) # Reemplazar todos los valores por NaN


    # Eliminamos filas fijandonos en las columnas del quality flag

    # Nos quedamos con valores de up_flag = 0, que son los de mejor calidad, los demás los marcamos como nulos
    mask_invalid_up = df_temp['up_flag'] != 0
    df_temp.loc[mask_invalid_up,'up_flag'] = np.nan
    # Lo mismo para down_flag
    mask_invalid_down = df_temp['down_flag'] != 0
    df_temp.loc[mask_invalid_down,'down_flag'] = np.nan


    # Eliminamos filas donde ambas flags son nulas
    df_temp.dropna(subset=['up_flag','down_flag'], how='all', inplace=True)


    # Eliminamos caracteres no numericos en las columnas numéricas que impidan convertirlas a float

    # Primero definimos las columnas numéricas    
    col_numericas = ['LAI_Miller_up', 'LAI_Warren_up', 'LAIe_Miller_up', 'LAIe_Warren_up', 'LAI_Miller_down', 
                    'LAI_Warren_down', 'LAIe_Miller_down', 'LAIe_Warren_down', 'LAI_Miller_up_err', 
                    'LAI_Warren_up_err', 'LAIe_Miller_up_err', 'LAIe_Warren_up_err', 'clumping_Miller_up', 
                    'clumping_Warren_up', 'LAI_Miller_down_err', 'LAI_Warren_down_err', 'LAIe_Miller_down_err', 
                    'LAIe_Warren_down_err', 'clumping_Miller_down', 'clumping_Warren_down', 'clumping_Miller_up_err', 
                    'clumping_Warren_up_err', 'clumping_Miller_down_err', 'clumping_Warren_down_err']

    # Removemos caracteres no numericos y forzamos el tipo de datos
    for col in col_numericas:
        if df_temp[col].dtypes != 'float':
            #print(col)
            # Limpiamos las columnas numericas de caracteres indeseados.
            df_temp[col] = df_temp[col].str.strip().str.replace('(', '').str.replace('(', '').str.replace(')', '')
            #df_temp[col].astype(float, errors='ignore')
            df_temp[col] =  pd.to_numeric(df_temp[col],errors='coerce')


    # Eliminamos filas con errores más altos de clumpling
    df_temp.loc[df_temp['clumping_Warren_up_err'] >= 0.1,'LAI_Warren_up'] = np.nan
    df_temp.loc[df_temp['clumping_Warren_down_err'] >= 0.35, 'LAI_Warren_down'] = np.nan


    # Definimos las **quality flags** especificas para las medidas del método Warren ('Warren_up_flag','Warren_down_flag').  
    #Nos basamos en las quality flags generales ('up_flag' y 'down_flag').
    df_temp.loc[df_temp['LAI_Warren_up'].notnull() & df_temp['LAI_Warren_down'].notnull() & (df_temp['up_flag'] == 0) & (df_temp['down_flag'] == 0), ['Warren_up_flag','Warren_down_flag']] = 0

    df_temp.loc[df_temp['LAI_Warren_up'].notnull() & df_temp['LAI_Warren_down'].isnull() & (df_temp['up_flag'] == 0) & (df_temp['down_flag'] == 0), 'Warren_down_flag'] = np.nan
    df_temp.loc[df_temp['LAI_Warren_up'].notnull() & df_temp['LAI_Warren_down'].isnull() & (df_temp['up_flag'] == 0) & (df_temp['down_flag'] == 0), 'Warren_up_flag'] = 0

    df_temp.loc[df_temp['LAI_Warren_up'].isnull() & df_temp['LAI_Warren_down'].notnull() & (df_temp['up_flag'] == 0) & (df_temp['down_flag'] == 0), 'Warren_up_flag'] = np.nan
    df_temp.loc[df_temp['LAI_Warren_up'].isnull() & df_temp['LAI_Warren_down'].notnull() & (df_temp['up_flag'] == 0) & (df_temp['down_flag'] == 0), 'Warren_down_flag'] = 0

    df_temp.loc[(df_temp['up_flag'] == 0) & df_temp['down_flag'].isnull(), 'Warren_down_flag'] = np.nan
    df_temp.loc[df_temp['LAI_Warren_up'].notnull() & (df_temp['up_flag'] == 0) & df_temp['down_flag'].isnull(), 'Warren_up_flag'] = 0

    df_temp.loc[df_temp['up_flag'].isnull() & (df_temp['down_flag'] == 0), 'Warren_up_flag'] = np.nan
    df_temp.loc[df_temp['LAI_Warren_down'].notnull() & df_temp['up_flag'].isnull() & df_temp['down_flag'] == 0, 'Warren_down_flag'] = 0


    # **Calculamos el LAI y su error** en base a las quality flags. 
    df_temp.loc[(df_temp['Warren_up_flag'] == 0) & (df_temp['Warren_down_flag'] == 0), 'LAI_Warren'] = df_temp['LAI_Warren_up'] + df_temp['LAI_Warren_down']
    df_temp.loc[(df_temp['Warren_up_flag'] == 0) & (df_temp['Warren_down_flag'] == 0), 'LAI_Warren_err'] = df_temp['LAI_Warren_up_err'] + df_temp['LAI_Warren_down_err'] 

    df_temp.loc[(df_temp['Warren_up_flag'] == 0) & df_temp['Warren_down_flag'].isnull(), 'LAI_Warren'] = df_temp['LAI_Warren_up']
    df_temp.loc[(df_temp['Warren_up_flag'] == 0) & df_temp['Warren_down_flag'].isnull(), 'LAI_Warren_err'] = df_temp['LAI_Warren_up_err']

    df_temp.loc[df_temp['Warren_up_flag'].isnull() & (df_temp['Warren_down_flag'] == 0), 'LAI_Warren'] = df_temp['LAI_Warren_down']
    df_temp.loc[df_temp['Warren_up_flag'].isnull() & (df_temp['Warren_down_flag'] == 0), 'LAI_Warren_err'] = df_temp['LAI_Warren_down_err']


    # **Unificamos** en una sola clase 'Evergreen Needleleaf' y 'Evergreen Needleleaf Forest'.  
    # También 'Deciduous Broadleaf' y 'Deciduous Broadleaf Forest'.
    df_temp.loc[df_temp['IGBP_class'] == 'Evergreen Needleleaf Forest', 'IGBP_class'] = 'Evergreen Needleleaf'
    df_temp.loc[df_temp['IGBP_class'] == 'Deciduous Broadleaf Forest', 'IGBP_class'] = 'Deciduous Broadleaf'


    # Nos quedamos con el LAI de las clases **forest** solo cuando tiene ambas medidas (LAI up y LAI down).

    # Primero definimos las clases que forman parte de forest. 
    class_forest = ['Mixed Forest', 'Evergreen Needleleaf', 'Evergreen Broadleaf', 'Deciduous Broadleaf']

    # Descartamos filas cuando falta la medida de up
    df_temp = df_temp.loc[~((df_temp['Warren_up_flag'] == 0) & df_temp['Warren_down_flag'].isnull() & (df_temp['IGBP_class'].isin(class_forest)))]
    # Descartamos filas cuando falta la medida de down
    df_temp = df_temp.loc[~(df_temp['Warren_up_flag'].isnull() & (df_temp['Warren_down_flag'] == 0) & (df_temp['IGBP_class'].isin(class_forest)))]


    # Eliminamos filas con **errores** más altos de LAI.
    df_temp = df_temp[df_temp['LAI_Warren_err'] <= 0.45]


    # Eliminamos las filas donde el LAI es nulo
    df_temp.dropna(subset=['LAI_Warren'], inplace=True)
    len(df_temp)


    # Eliminamos **valores anómalos** de LAI por cobertura.
    lai_max_cover = {
        'Mixed Forest': 7.5,
        'Evergreen Needleleaf': 8.0,
        'Open Shrublands': 5.0,
        'Croplands': 6.0,
        'Grasslands': 5.0,
        'Evergreen Broadleaf': 7.0,
        'Closed Shrublands': 5.0,
        'Deciduous Broadleaf': 7.0
    }

    for land_cover, lai_max in lai_max_cover.items():
        df_temp = df_temp[~((df_temp['IGBP_class'] == land_cover) & (df_temp['LAI_Warren'] > lai_max))]


    # Eliminamos **valores anómalos** de LAI según el site.
    lai_max_site ={
    'Harvard Forest':6.0,
    'Jones Ecological Research Center':5.5,
    'Jornada':0.04,
    'Konza Prairie Biological Station':3.0,
    'Lajas Experimental Station': 2.5,
    'Lenoir Landing':7.0,
    'Moab':0.2,
    'Niwot Ridge Mountain Research Station':5.0,
    'Onaqui Ault':0.2,
    'Oak Ridge':7.5,
    'Ordway Swisher Biological Station':2.0,
    'Pu u Maka ala Natural Area Reserve':7.0,
    'Smithsonian Conservation Biology Institute':8.0,
    'Smithsonian Environmental Research Center':6.0,
    'Soaproot Saddle':2.0,
    'Santa Rita':0.3,
    'Steigerwaldt Land Services':7.0,
    'North Sterling':0.6,
    'Talladega National Forest':6.0,
    'Lower Teakettle':5.5,
    'University of Kansas Field Site':7.0,
    'Underc':6.5,
    'Woodworth':2.0,
    'Bartlett Experimental Forest':6.5,
    'Blandy Experimental Farm':7.0,
    'Lyndon B. Johnson National Grassland':6.0,
    'Central Plains Experimental Range':0.25,
    'Dead Lake':7.0,
    'Disney Wilderness Preserve':1.5}

    for site, lai_max in lai_max_site.items():
        df_temp = df_temp[~((df_temp['Site'] == site) & (df_temp['LAI_Warren'] > lai_max))]

    lai_min_site ={
    'Harvard Forest':2.2,
    'Lenoir Landing':2.0,
    'Oak Ridge':2.5,
    'Bartlett Experimental Forest':3.0,
    'Dead Lake':0.6}

    for site, lai_min in lai_min_site.items():
        df_temp = df_temp[~((df_temp['Site'] == site) & (df_temp['LAI_Warren'] < lai_min))]


    # Quitamos el **dato UTC** a la columna fecha para estandarizarla con el dataset de satelite.
    df_temp['TIME_IS'] = pd.to_datetime(df_temp['TIME_IS']).dt.tz_localize(None)

    return df_temp



def merge_datasets(df_sat_temp, df_insitu_temp):

    # **Merge de data sets**. Para cada site identificamos las medidas insitu mas cercanas a la fecha de adquisicion del satélite
    df_list = []
    days_diff_max = 5 # Diferencia maxima (en días) entre la medida de satelite y la medida in situ

    # Itera sobre cada coordenada
    for site, lat, lon in coords:
        # Filtra el DataFrame de df_insitu y df_sat según el 'Site'
        df_insitu_site = df_insitu_temp[df_insitu_temp['Site'] == site].copy()
        df_sat_site = df_sat_temp[df_sat_temp['Site_sat'] == site].copy()

        rows = []

        # Itera sobre los datos de satélite y sobre los datos in situ
        for index_sat, date_sat in df_sat_site['time'].items():
            for index_insitu, date_insitu in df_insitu_site['TIME_IS'].items():
                # Calcula la diferencia en días entre las fechas
                dif = (date_insitu - date_sat) / np.timedelta64(1, 'D')

                if abs(dif) <= days_diff_max: # Si la diferencia en dias es menor...
                    # Crea una lista con la fila combinada de ambas fuentes de datos
                    row = list(df_sat_site.loc[index_sat]) + list(df_insitu_site.loc[index_insitu])
                    rows.append(row)

        # Si se encontraron filas, procesa los datos
        if rows:
            # Crear el DataFrame con las filas acumuladas
            df_cross_site = pd.DataFrame(rows, columns=list(df_sat_site.columns) + list(df_insitu_site.columns))

            # Añadir una columna 'delta' con la diferencia en tiempo entre el satélite y las mediciones in situ
            df_cross_site['delta'] = abs(df_cross_site['time'] - df_cross_site['TIME_IS'])

            # Añadir las fechas separadas de 'date_sat' y 'date_insitu'
            df_cross_site['date_sat'] = df_cross_site['time'].dt.date
            df_cross_site['date_insitu'] = df_cross_site['TIME_IS'].dt.date

            # Ordenar y eliminar duplicados por 'date_insitu'
            df_cross_site = df_cross_site.sort_values(by=['date_insitu', 'delta']).drop_duplicates(subset='date_insitu', keep='first')

            # Ordenar y eliminar duplicados por 'date_sat'
            df_cross_site = df_cross_site.sort_values(by=['date_sat', 'delta']).drop_duplicates(subset='date_sat', keep='first')


            # Agregar el DataFrame de este sitio a la lista
            df_list.append(df_cross_site)

    # Combinar todos los DataFrames en uno solo
    df_temp = pd.concat(df_list, axis=0, ignore_index=True)

    return df_temp