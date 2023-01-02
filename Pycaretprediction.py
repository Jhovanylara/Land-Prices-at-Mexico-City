import sys
sys.path.append('..')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
import plotly.express as px
import urbanpy as up
import sklearn.metrics as metrics
from geopy import Nominatim
from matplotlib import rcParams
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from ipywidgets import interact, Dropdown, FloatSlider, IntSlider
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from pycaret.regression import *

df=pd.read_csv('https://raw.githubusercontent.com/Jhovanylara/Land-Prices-at-Mexico-City/master/Datasets/ETL.csv') #Leemos csv luego del ETL

missing_cost=df.loc[df['target']==0] #Hacemos un dataframe solo con los costos vacíos
df['target'].replace(0,np.nan, inplace=True) #Reemplazamos por NaN para luego borrarlos con la función dropna
df=df.dropna(subset="target") #Eliminamos las filas con datos faltantes

#Quitamos los outliers

#Obtenemos df sin colonia ni Alcaldia
X=df.iloc[:,2:]

list=[]
for col in X.columns:
    list.append(col)
list.remove('lat')
list.remove('lon')

df_Limpieza = X


filtro1 = np.array([True] * len(df_Limpieza))

for col in list:
    zscore = abs(stats.zscore(df_Limpieza[col])) 
    filtro1 = (zscore < 2) & filtro1  #PONEMOS UN FILTRO DE ZSCORE >= 2
    
df_SinOutliers = df_Limpieza[filtro1] 

Z=df.iloc[:,2:]
demo = setup(data = Z, target = 'target', 
                   normalize = True,
                   #transformation= True, transformation_method = 'yeo-johnson', 
                   #transform_target = True, remove_outliers= True,
                   #remove_multicollinearity = True,
                   ignore_low_variance = True, combine_rare_levels = True) 

#Seleccionamos los tres mejores modelos
Random_forest = create_model('rf')
Gradient_boosting = create_model('gbr')
Bayesian_ridge = create_model('br')
#Los afinamos
Random_forest = tune_model(Random_forest)
Gradient_boosting = tune_model(Gradient_boosting)
Bayesian_ridge = tune_model(Bayesian_ridge)

# Blending models
blender = blend_models(estimator_list = [Random_forest, Gradient_boosting, Bayesian_ridge])
# Creamos modelo para predicciones
model = finalize_model(blender)


#Completamos costos faltantes en missing_cost

missing=missing_cost.iloc[:,2:] #Quitamos las columnas Colonia y Alcaldía que no forman parte del train
r=len(missing)
n=678
for i in range(0,r):
    #Creamos un dataframe con la fila a predecir
    a=missing.iloc[i]
    df22=pd.DataFrame(a)
    df22=df22.T
    df22=df22.drop(['target'], axis=1)
    
    prediction = model.predict(df22).round(2)
    if prediction[0] < 1000:
        prediction=np.array([1272.79])
    missing.at[n,'target']=prediction[0]
    n+=1

missing_cost['target']=missing['target']

df=pd.concat([df,missing_cost]) #Concatenamos df con los valores predecidos

df.to_csv('PycaretComplete.csv', index=False) #Guardamos en un nuevo csv


#Creamos función para predecir
def Pycaret_predict_price(latitude, longitud):

    mx=up.download.nominatim_osm('Ciudad de Mexico, Mexico')
    cdmx=pd.DataFrame({'lat':latitude,'lon':longitud}, index=[0])
    
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
        cdmx[f'{i}_dist(km)']=dist_up
    #return cdmx
    prediction = model.predict(cdmx).round(2)
    return prediction