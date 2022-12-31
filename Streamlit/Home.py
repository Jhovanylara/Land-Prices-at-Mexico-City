import streamlit as st
import snowflake.connector
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
from prediction import predict_price
from geopy import Nominatim


st.title("TRULLY CHALLENGE")

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
    se seleccionaran 4 tipos de datos de lugares: alimentos, salud, educación, servicios financieros, 
    y se tomará la distancia en vuelo de pájaro para cada uno de los lugares a la coordenada base de predicción.'''
with d2:
    st.subheader("Fuera de alcance")
    '''Se obtendrán unicamente datos geográficos de la ciudad de México como variables, no se está considerando el tiempo
    de desplazamiento hacia los lugares seleccionados. Se descarta cualquier coordenada fuera de los límites de la CDMX '''

st.write('***')

h1,h2,h3=st.columns(3)
with h2:
    st.header("Metodology")
'''
The work is based on a quantitative approach which will help to establish relations between quantitative variables from socioeconomic and health indicators, such as GDP per capita,
public spending on education, under-5 mortality rate, % of rural population, % of poverty, % of total
health spending etc. which can help to generate a context of the possible influence of each variable in
life expectancy at birth.

We have worked on the development of a ML algorithm that allows us to generate future predictions using
expectations of these indicators.'''



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

st.header("Calcula el costo/m2 en un punto de la CDMX")
point=st.text_input('Escribe una coordenada con formato: lat, lon', placeholder='19.413464, -99.135515')
a=within_CDMX(point)
if a==False:
    '''La coordenada dada, no está dentro en CDMX'''
else:
    a=predict_price(19.413464, -99.135515)
    f'''El costo del m2 es: {a}'''