import sys
import streamlit as st
import pandas as pd
#from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
import geopy as geopy
from geopy import Nominatim
import urbanpy as up
import matplotlib.pyplot as plt
import urbanpy as up
from pycaret.regression import *
from prediction import *
#from Pycaretprediction import *

col1, col2, col3=st.columns(3)
with col2:
    st.image('https://raw.githubusercontent.com/Jhovanylara/Land-Prices-at-Mexico-City/master/Images/logo-color.svg')

st.title("TRULLY CHALLENGE CDMX")
st.image('https://raw.githubusercontent.com/Jhovanylara/Land-Prices-at-Mexico-City/master/Images/BELLAS_ARTES.jpg', use_column_width=True)

st.header("Planteamiento del problema")
'''
A partir de un dataset en csv con datos de la Ciudad de México(CDMX), se nos solicita completar los datos
faltantes en la columna 'Alcaldía' y en la columna de 'costo/m2'. Además se requiere enriqucer
los datos con variables que pudieran explicar el costo del m2 en la CDMX; para luego poder hacer un modelo
que prediga el costo promedio en las Alcaldías faltantes.  

Por último se solicita predecir el costo/m2 a partir de un input de coordenadas de la cdmx.
'''

d1,d2=st.columns(2)
with d1:
    st.subheader("Alcance")
    '''Obtener documentación pertinente para explicar el costo/m2 dentro de la ciudad de México, 
    se seleccionaran 4 categorías de sitios de interes: alimentos, salud, educación, servicios financieros, 
    y se tomará la distancia en vuelo de pájaro para cada uno de los sitios a la coordenada base de predicción.'''
with d2:
    st.subheader("Fuera de alcance")
    '''Se obtendrán unicamente datos geográficos de la ciudad de México como variables, no se está considerando el tiempo
    de desplazamiento hacia los lugares seleccionados. Se descarta cualquier coordenada fuera de los límites de la CDMX '''

st.write('***')

h1,h2,h3=st.columns(3)
with h2:
    st.header("Metodología")
'''
Primero se completó la columna Alcaldía del df original a partir de los datos de geolocalización de las columnas de latitud
y longitud. Gracias a la función geolocator, se puede obtener la dirección completa a partir de las coordenadas
y así podemos saber a que alcaldía pertenece cada punto a geolocalizar.

Luego se procesaron los Outliers en la columna objetivo con un ZScore>=2.5 para así eliminar 8 elementos que pudieran
generar ruido en la predicción.

Se eligieron 4 categorías de sitios de interes que al estar más cercanos a un terreno, pudieran influir en el costo del m2 y se tomó la distancia a cada sitio de la categoría, para luego incluirlo al dataframe original. 

    - Alimentación: Tiendas de conveniencia, carnicerías, supermercados, verdulerías, malls
    - Salud: Hospitales, farmacias, clinicas, doctores
    - Educación: Escuelas, Universidades, kinders
    - Finanzas: Bancos, cajeros automáticos
'''
tab1, tab2=st.tabs(['Dataframe crudo', 'Dataframe con ETL'])
with tab1:
    'Dataframe crudo'
    df=pd.read_excel('https://github.com/Jhovanylara/Land-Prices-at-Mexico-City/blob/master/Datasets/Challenge_CDMX_Mapa(preciom2).xlsx?raw=true')
    st.dataframe(df)
with tab2:
    'Dataframe enriquecido'
    st.text('La columna target tiene aún datos vacíos, que luego predeciremos')
    df1=pd.read_csv('https://raw.githubusercontent.com/Jhovanylara/Land-Prices-at-Mexico-City/master/Datasets/ETL.csv')
    st.dataframe(df1)
st.write('***')
st.header("Entrenamiento")
'''
Para la predicción se limpiaron los Outliers, se eliminaron columnas redundantes y se entrenó con el modelo HuberRegressor. Luego se hizo la predicción
para cada elemento faltante en la columna target del dataframe, para quedar de la siguiente manera:
'''
mx=up.download.nominatim_osm('Ciudad de Mexico, Mexico')

df3=pd.read_csv('https://raw.githubusercontent.com/Jhovanylara/Land-Prices-at-Mexico-City/master/Datasets/CDMXcomplete.csv')
st.dataframe(df3)


st.subheader('Visualizamos las zonas coloreando por precio')
fig = px.scatter_mapbox(
    df3,
    lat = "lat",
    lon = "lon",
    color = "target",
    width = 900,
    height = 600,
    hover_data = ["Colonia", "target"]
    );

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)
st.write('***')
'''
Ya con el dataframe listo, finalmente podemos hacer una predicción a partir de las coordenadas dadas. Se limitó a que solo se haga el cálculo cuando el punto ingresado se encuentra dentro de la CDMX.
Se construyó una función que al obtener unas coordenadas válidas, obtiene los datos de las 4 categorías y calcula la distancia de cada sitio al punto a buscar y lo ingresa a un nuevo dataframe, el cuál luego será usado, para hacer una predicción del costo/m2 a partir de todos los datos recolectados.
'''
def within_CDMX(lat,lon):
    coord=(lat, lon)
    geolocator=Nominatim(user_agent='test/1')
    location=geolocator.reverse(coord) #Aquí ya tenemos la dirección completa
    strloc=str(location) #cambiamos el tipo de dato a string
    dentro=False
    if 'Ciudad de México' in strloc:
        dentro=True
    if 'Ciudad de Mexico' in strloc:
        dentro=True
    return dentro

def validate(in_):
    
    try:
        a, b = in_.split(", ")
        _, _ = str(a), str(b)
    except Exception:
        return False
    else:
        return True

st.header("Calcula el costo/m2 en un punto de la CDMX")
point=st.text_input('Escribe una coordenada con formato: latitud, longitud', placeholder='19.413464, -99.135515')

if point=="":
        "Sin entrada"
elif validate(point)==False:
    "Formato inválido"
else:
    
    s=point.split(",")
    latitud=s[0]
    longitud=s[1]
    a=within_CDMX(latitud, longitud)
    if a==False:
        '''La coordenada dada, no está en CDMX'''
    else:
        latitud=float(latitud)
        longitud=float(longitud)
        longitud=longitud.strip()
        costom2=predict_price(latitud, longitud)
        f'''El costo del m2 es: ${costom2[0]} MXN'''
        
st.write('***')
'''
Los datasets y los notebooks se encuentran en el repositorio de github: https://github.com/Jhovanylara/Land-Prices-at-Mexico-City 
'''

'Autor: Jhovany Lara Nava'