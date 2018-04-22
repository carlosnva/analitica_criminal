#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


# CARACTÉRES ESPECIALES ESPAÑOL
# Á - \u00c1
# É - \u00c9
# Í - \u00cd
# Ó - \u00d3
# Ú - \u00da
# Ñ - \u00d1
# á - \u00e1
# é - \u00e9
# í - \u00ed
# ó - \u00f3
# ú - \u00fa
# ñ - \u00f1
# ¡   \u00a1
# ¿   \u00bf

# Importar datos y funciones básicas
import pandas as pd
import numpy as np
import scipy
import math
# Importar librerías gráficas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Importar librerías de formato
from datetime import tzinfo, timedelta, datetime
# Importar modelos de aprendizaje
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Primera prueba:

    1) Conocer cómo está compuesta la base de datos
"""

df_train = pd.read_csv("./input-data/chicago_train.csv",index_col=None)
print(df_train.dtypes)

#input("Presiona una tecla para continuar...(I)")

"""
Conclusiones:
    1) Hay que predecir la categoría
    2) Fechas absolutas, dividir según caso (día, mes, año)
    3) Los distritos sólo son útiles en caso de georreferencia
    4) La dirección es igual
    5) X y Y son datos en latitud y logitud

Segunda prueba:

    2) Conocer cómo está compuesto el crimen a lo largo del año
"""

train = pd.read_csv('./input-data/chicago_train.csv', parse_dates=['DATE'])

train['DayOfYear'] = train['DATE'].map(lambda x: x.strftime("%m-%d"))
train['DayOfWeek'] = train['DATE'].map(lambda x: x.strftime("%A"))
train['MonthOfYear'] = train['DATE'].map(lambda x: x.strftime("%m"))
train['TimeOfYear'] = train['DATE'].map(lambda x: x.strftime("%H"))

df_global = train[['CATEGORY','DayOfYear']].groupby(['DayOfYear']).count()
df_global.plot(y='CATEGORY', label='N\u00famero de eventos', figsize=(6,4)) 
plt.title("Patrones criminales")
plt.ylabel('N\u00famero de cr\u00edmenes')
plt.xlabel('D\u00eda del a\u00f1o')
plt.grid(True)
plt.savefig('./output-data/Distribution_of_Crimes_by_Day_Year.png')

plt.show()
plt.close()

df_global = train[['CATEGORY','DayOfWeek']].groupby(['DayOfWeek']).count()
df_global.plot(y='CATEGORY', label='N\u00famero de eventos', figsize=(6,4)) 
plt.title("Patrones criminales")
plt.ylabel('N\u00famero de cr\u00edmenes')
plt.xlabel('D\u00eda de la semana')
plt.grid(True)
plt.savefig('./output-data/Distribution_of_Crimes_by_Day_Week.png')

plt.show()
plt.close()

df_global = train[['CATEGORY','MonthOfYear']].groupby(['MonthOfYear']).count()
df_global.plot(y='CATEGORY', label='N\u00famero de eventos', figsize=(6,4)) 
plt.title("Patrones criminales")
plt.ylabel('N\u00famero de cr\u00edmenes')
plt.xlabel('Mes del a\u00f1o')
plt.grid(True)
plt.savefig('./output-data/Distribution_of_Crimes_by_Month_Year.png')

plt.show()
plt.close()

df_global = train[['CATEGORY','TimeOfYear']].groupby(['TimeOfYear']).count()
df_global.plot(y='CATEGORY', label='N\u00famero de eventos', figsize=(6,4)) 
plt.title("Patrones criminales")
plt.ylabel('N\u00famero de cr\u00edmenes')
plt.xlabel('Hora del d\u00eda')
plt.grid(True)
plt.savefig('./output-data/Distribution_of_Crimes_by_Hour_Year.png')

plt.show()
plt.close()

"""
Conclusiones:
    1) El primer patrón que presenta el crimen es estacional y dos veces por
    mes.

Hipótesis:
    1) La primera quincena (mitad de mes) y la segunda quincena (fin de mes)
    está asociado a tipos de crímenes
    2) Los días de paga están más unidos a ciertos crímenes

Tercera prueba:

    3) Conocer cómo está compuesto el crimen en sus categorías
"""

Crime_Categories = list(df_train.loc[:,'CATEGORY'].unique())
print("N\u00famero de categor\u00edas: " + str(len(Crime_Categories)))
for crime in Crime_Categories:
    print(crime)
number_of_crimes = df_train.CATEGORY.value_counts()

n_crime_plot = sns.barplot(x=number_of_crimes.index,y=number_of_crimes)
n_crime_plot.set_xticklabels(number_of_crimes.index,rotation=90)
n_crime_plot.set(ylabel='Valor acumulado de la muestra')
fig = n_crime_plot.get_figure()
fig.savefig("./output-data/Composici\u00f3n del crimen en CHI.png",bbox_inches='tight')
plt.show()
plt.close(fig)

relative_crime = number_of_crimes / sum(number_of_crimes)
relative_crime = relative_crime.cumsum()
r_crime_plot = sns.tsplot(data=relative_crime)
r_crime_plot.set_xticklabels(relative_crime.index,rotation=90)
r_crime_plot.set_xticks(np.arange(len(relative_crime)))
fig = r_crime_plot.get_figure()
fig.savefig("./output-data/Composici\u00f3n porcentual del crimen en CHI.png",bbox_inches='tight')

plt.show()
plt.close(fig)

SubCrime_Categories = list(relative_crime[0:12].index)
print("Las siguientes categorias:")
print(SubCrime_Categories)
print("constituyen el {:.2%} del total de los crimenes".format(relative_crime[12]))

#input("Presiona una tecla para continuar...(III)")

day_week = {
    'Monday':0,
    'Tuesday':1,
    'Wednesday':2,
    'Thursday':3,
    'Friday':4,
    'Saturday':5,
    'Sunday':6
}

df_train['DayOfWeek'] = train['DayOfWeek']

df_train['DOW']=df_train.DayOfWeek.map(day_week)
df_train['hour']=pd.to_datetime(df_train.DATE).dt.hour

plt.figure(1,figsize=(6,4))
plt.hist2d(
    df_train.hour.values,
    df_train.DOW.values,
    bins=[24,7],
    range=[[-0.5,23.5],[-0.5,6.5]]#,
#    vmin = 0,
#    vmax = 1
)
plt.xticks(np.arange(0,24,6))
plt.xlabel('Time of Day')
plt.yticks(np.arange(0,7),['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.ylabel('Day of Week')
plt.gca().invert_yaxis()
plt.title('Occurance by Time and Day - All Categories')
plt.grid(False)
plt.show()
plt.close()

#############################################################################
#print(df_train['DOW'].head(25))
#print(df_train['DOW'].describe())
#print(df_train['hour'].describe())
#
#
#
#heat_set = pd.DataFrame({'DOW' : df_train['DOW'], 'hour' : df_train['hour']})
#sns.heatmap(heat_set, annot=True,xticklabels=np.arange(0,24,6),yticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']);
#
input("Presiona una tecla para continuar...(IV)")
#############################################################################


#latitude and longitude of map data
# 41.644604096	-87.928909442
# 42.022671246	-87.524529378
#
#

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

drive_in = './input-data/'
drive_out = './output-data/'
train = pd.read_csv(drive_in+'sf_train_fixed.csv')

#get a unique list of categories
cats = list(set(train.Category))
mapdata = np.loadtxt(drive_in+"sf_map_copyright_openstreetmap.txt")

#turn strings into dates
dates = []
datesAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            for date in train.Dates])

#set up pandas
startDate = (np.min(datesAll)).date()
endDate = (np.max(datesAll)).date()
alldates = pd.bdate_range(startDate, endDate, freq="m")
dayDF = pd.DataFrame(np.NAN, index=alldates, columns=['x'])

#pLoop = 1
for cat in cats:
    saveFile = cat+'.png'
    if cat in SubCrime_Categories:
        fig = plt.figure(figsize = (11.69, 8.27))
        plt.title(cat)
        
        #plot image
        ax = plt.subplot(2,2,1)
        ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box)
    
        Xcoord = (train[train.Category==cat].X).values
        Ycoord = (train[train.Category==cat].Y).values
        dates = datesAll[np.where(train.Category==cat)]
        Z = np.ones([len(Xcoord),1])
            
        #create dataframe
        df = pd.DataFrame([ [ Z[row][0],Xcoord[row],Ycoord[row]  ] for row in range(len(Z))],
               index=[dates[row] for row in range(len(dates))],
               columns=['z','xcoord','ycoord']) 
         
        #resample to sum by month
        #df2 = df.resample('m', how='sum')
        df2 = df.resample('m').sum()
        
        #kde plot by year
        kdeMaxX = []
        kdeMaxY = []
        for yLoop in range(2003,2015):
            allData2 = df[(df.index.year == yLoop)]
           
            kde = scipy.stats.gaussian_kde(np.array(allData2['xcoord']))
            density = kde(np.array(allData2['xcoord']))
            kde2 = scipy.stats.gaussian_kde(np.array(allData2['ycoord']))
            density2 = kde2(np.array(allData2['ycoord']))
            kdeMaxX.append((allData2['xcoord'][density==np.max(density)]).values[0])
            kdeMaxY.append((allData2['ycoord'][density2==np.max(density2)]).values[0])
            
        
        #create a quiver plot to show movement of centre of KDE per year
        kdeOut = sns.kdeplot(np.array(allData2['xcoord']), np.array(allData2['ycoord']),shade=True, cut=10, clip=clipsize,alpha=0.5)
        kdeMaxX = np.array(kdeMaxX)
        kdeMaxY = np.array(kdeMaxY)
        plt.quiver(kdeMaxX[:-1], kdeMaxY[:-1], kdeMaxX[1:]-kdeMaxX[:-1], kdeMaxY[1:]-kdeMaxY[:-1], scale_units='xy', angles='xy', scale=1,color='r')
  
        #create uniform time series
        allTimes = dayDF \
        .join(df2) \
        .drop('x', axis=1) \
        .fillna(0)
    
        #movAv = pd.rolling_mean(allTimes['z'],window=12,min_periods=1)
        movAv = pd.Series(allTimes['z']).rolling(window=12,min_periods=1).mean()
    
        #time series plot with 12 month moving average
        ax = plt.subplot(2,1,2)
        plt.plot(allTimes.index,allTimes['z'])
        plt.plot(allTimes.index,movAv,'r')
        
        #heatmap to look how data varies by day of week
        ax = plt.subplot(2,2,2)
        heatData = []
        yLoopCount=0
        weekName = ['mon','tue','wed','thu','fri','sat','sun']
        yearName = ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']
        for yLoop in range(2003,2015):
            heatData.append([])
            for dLoop in range(7):
                allData2 = df[(df.index.year == yLoop) & (df.index.weekday == dLoop)]
                heatData[yLoopCount].append(sum(allData2['z'].values))
            yLoopCount+=1
        
        #normlise
        heatData = np.array(heatData)/np.max(np.array(heatData))
        sns.heatmap(heatData, annot=True,xticklabels=weekName,yticklabels=yearName);
        
        plt.title(cat)
        plt.savefig(drive_out+saveFile)
        print("La categoria {0} es relevante".format(cat))

    else:
        print("La categoria {0} no es relevante".format(cat))
        #input("Presiona una tecla para continuar...(V)")        

#pLoop+=1

print("Fin del programa")

