import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geopy import Nominatim
from matplotlib import rcParams
import scipy.stats as stats
import plotly.express as px
import urbanpy as up
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm

#---------------------------#
# Ingestamos tabla original #
#---------------------------#
datos=pd.read_excel('Challenge CDMX Mapa (precio m2).xlsx')
df=pd.DataFrame(datos)

df['Alcaldía']=df['Alcaldía'].replace(["*"], 0)#Reemplazamos el símbolo * por zeros
missing_alcaldias=df.loc[df['Alcaldía']==0] #creamos un dataframe con las filas en zero

dfi=missing_alcaldias.reset_index()
indice=dfi['index'].tolist() #creamos una lista con el indice de datos faltantes
dfi.rename(columns={'index':'lista'})  #Renombramos columna index por lista

#------------------------------------------------------#
# Utlizamos Geolocator para completar datos faltantes  #
#------------------------------------------------------#
'''Podemos saber la dirección con coordenadas con la función geolocator y luego 
asignar la alcaldía de esa dirección a la columna "Alcaldía" que tiene datos faltantes'''

for i in indice:
    #Obtenemos la dirección a partir de las coordenadas
    latitud=missing_alcaldias.loc[i,'Latitude (generated)']
    longitud=missing_alcaldias.loc[i,'Longitude (generated)']
    coord=(latitud, longitud)
    geolocator=Nominatim(user_agent='test/1')
    location=geolocator.reverse(f'{latitud},{longitud}') #Aquí ya tenemos la dirección completa
    strloc=str(location) #cambiamos el tipo de dato a string
    if i==126 or i==270 or i==640:
        a=strloc.split(',')[-3] #Dividimos la direccion para obtener solo el dato de la alcaldía (estos datos no tienen CP)
    elif i==666:
        a='IZTAPALAPA'
    elif i==37:
        a='BENITO JUAREZ'
    else:
        a=strloc.split(',')[-4] #Dividimos la direccion para obtener solo el dato de la alcaldía
    a=a.strip().upper() #quitamos el espacio y cambiamos a mayúsculas
    df.loc[df.index==i,'Alcaldía']=a #Asignamos la alcaldía que este en el indice

#-------------------------------------#
# Quitamos Outliers sin tocar los NaN #
#-------------------------------------#

#Hacemos un dataframe con las filas sin datos en columna costo_m2
df['Avg. costo_m2_terreno']=df['Avg. costo_m2_terreno'].fillna(0)
missing_cost=df[df['Avg. costo_m2_terreno']==0]

df['Avg. costo_m2_terreno'].replace(0,np.nan, inplace=True) #Reemplazamos por NaN
df=df.dropna(subset="Avg. costo_m2_terreno") #Eliminamos las filas con datos faltantes

#Quitamos Outliers
col_num=['Avg. costo_m2_terreno']
filtro1 = np.array([True] * len(df))
for col in col_num:
    zscore = abs(stats.zscore(df[col])) 
    filtro1 = (zscore < 2.5) & filtro1  #PONEMOS UN FILTRO DE ZSCORE >= 2.5
df_SinOutliers = df[filtro1] 

df_SinOutliers=pd.concat([df_SinOutliers,missing_cost]) #Juntamos con dataframe sin datos en columna costo_m2

#-------------------#
#  Quitamos acentos #
#-------------------#

def acentos(s):
    replacements=(('Á','A'),('É','E'),('Í','I'),('Ó','O'),('U','U'))
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

for i in df_SinOutliers['Alcaldía']:
    df_SinOutliers.loc[df_SinOutliers['Alcaldía']==i, 'Alcaldía']=acentos(i)

#-------------------------------#
#                               #
#   Extraemos más datos que     #
#   expliquen el costo del m2   #
#         con *Urbanpy*         #
#                               #
#-------------------------------#

mx=up.download.nominatim_osm('Ciudad de Mexico, Mexico')
cdmx=df_SinOutliers
cdmx.rename(columns={'Latitude (generated)':'lat','Longitude (generated)':'lon'}, inplace=True) #renombramos las columnas de coordenadas

#Cargamos datos de salud
es=up.download.overpass_pois(bounds=mx.total_bounds, facilities='health')
health=['hospital', 'pharmacy','clinic', 'dentist', 'doctors']
for i in health:
    z=es[es['poi_type']==i]
    dist_up, ind_up=up.utils.nn_search(
        tree_features=z[['lat','lon']].values,#Puntos de interes
        query_features=cdmx[['lat','lon']].values,#coordenadas de cada colonia
        metric='haversine' #Metrica de distancia
    )
    cdmx[f'{i}_near']=ind_up
    cdmx[f'{i}_dist(km)']=dist_up

#Cargamos datos de alimentos
es=up.download.overpass_pois(bounds=mx.total_bounds, facilities='food')
food=['convenience', 'supermarket','butcher', 'marketplace', 'greengrocer','mall']

for i in food:
    z=es[es['poi_type']==i]
    dist_up, ind_up=up.utils.nn_search(
        tree_features=z[['lat','lon']].values,#Puntos de interes
        query_features=cdmx[['lat','lon']].values,#coordenadas de cada colonia
        metric='haversine' #Metrica de distancia
    )
    cdmx[f'{i}_near']=ind_up
    cdmx[f'{i}_dist(km)']=dist_up

#Cargamos datos de educación
es=up.download.overpass_pois(bounds=mx.total_bounds, facilities='education')
education=['school', 'university','kindergarten']

for i in education:
    z=es[es['poi_type']==i]
    dist_up, ind_up=up.utils.nn_search(
        tree_features=z[['lat','lon']].values,#Puntos de interes
        query_features=cdmx[['lat','lon']].values,#coordenadas de cada colonia
        metric='haversine' #Metrica de distancia
    )
    cdmx[f'{i}_near']=ind_up
    cdmx[f'{i}_dist(km)']=dist_up

#Cargamos datos financieros
es=up.download.overpass_pois(bounds=mx.total_bounds, facilities='finance')
finance=['bank', 'atm']

for i in finance:
    z=es[es['poi_type']==i]
    dist_up, ind_up=up.utils.nn_search(
        tree_features=z[['lat','lon']].values,#Puntos de interes
        query_features=cdmx[['lat','lon']].values,#coordenadas de cada colonia
        metric='haversine' #Metrica de distancia
    )
    cdmx[f'{i}_near']=ind_up
    cdmx[f'{i}_dist(km)']=dist_up


#----------------------------------------#
# Devolvemos un csv con los datos listos #
#----------------------------------------#

#Renombramos columna objetivo
cdmx.rename(columns={'Avg. costo_m2_terreno':'target'}, inplace=True)

#Cambiamos la posición al final
s=cdmx.pop('target')
cdmx=pd.concat([cdmx,s],1)

cdmx.to_csv('ETL.csv', index=False)