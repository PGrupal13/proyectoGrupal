![CO2](https://i.ibb.co/FYKNGrp/banner-co2-A.png)
# Consumo de Energia Global y CO2

##### Proyecto realizado para ![Henry](https://i.ibb.co/DgH9488/logoH.png) [Henry](https://github.com/soyHenry/proyecto_grupal_consumo_energ-a_co2/).     

##### Cojunto de datos utilizados:    
- [Emisiones CO2](https://drive.google.com/drive/folders/1nAoLHcmrtoDcDtgUT3UDzFPZZxKh8_1s)    
- [Global power plant database](https://drive.google.com/drive/folders/1nAoLHcmrtoDcDtgUT3UDzFPZZxKh8_1s)    
- [Energy consumption](https://drive.google.com/drive/folders/1nAoLHcmrtoDcDtgUT3UDzFPZZxKh8_1s)     

##### Visualización
Se implemento un dashboard que captura las métricas más importantes para el análisis usando Power BI. El acceso al mismo se proporciona a través de la plataforma de Streamlit usando el siguiente enlace: [StreamLit-App](https://pgrupal13-proyectogrupal-inicio-3klytr.streamlitapp.com/)

##### Aprendizaje automático
Las propuestas presentadas se enfocan en la predicción de las emisiones de CO2: usando diferentes enfoques. 

##### Almacenamiento y automatización
- El conjunto de datos basico ha sido almacenado en AWS: S3.   
- El acceso a los datos con sus respectivas modificaciones se lleva acabo desde AWS: Athena.
- La automatización se llevo a cabo usando la librería BOTO3 de python, la cuál permite conectar Python con los servicios antes mencionados.       
![Pipeline](https://i.ibb.co/c8hXvJw/pipeline.jpg)

##### Alteraciones a los datos 
- Se omitieron datos de regiones; conservando solo los datos reportados para países y algunos de sus territorios más detacados.     
- Se estandarizaron nombres de países y otros territorios.      
- Se construyeron subconjuntos de datos en base a [modelo relacional propuesto](https://github.com/PGrupal13/proyectoGrupal/blob/main/Info/DER.png). Para información adicional consultar [diccionario de datos](https://github.com/PGrupal13/proyectoGrupal/blob/main/Info/Diccionario.csv).    


##### Colaboradores
- **Alejandro Giraldo**(Ingeniero/Analista de datos): Desarrollo de KPI’s, automatización, vizualización de datos, análisis de datos, implementación de modelos de predicción usando aprendizaje automático.
- **Aurora Martínez**(Ingeniero de datos): Limpieza y organización de datos, Implementación de modelo de predicción usando aprendizaje automático.
- **Carlos Matich**(Analista de datos): Desarrollo de KPI’s, vizualización de datos, análisis de datos, implementación de modelo de predicción usando aprendizaje automático.
- **Ezequiel Carena**(Ingeniero de datos): Limpieza de datos, propuesta de modelo de predicción usando aprendizaje automático.
- **Horacio Guassardo**(Ingeniero/Analista de datos): Creación y carga de Data Warehouse, automatización, vizualización de datos, análisis de datos, implementación de modelo de predicción usando aprendizaje automático.