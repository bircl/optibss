# -*- coding: utf-8 -*-
# @Author  : Birkan Caliskan
# @Email   : birkanx@gmail.com

import geopandas as gpd
from os import listdir
import pandas as pd
import numpy as np
import pickle
from pyproj import Transformer
import os




filename = "d:\\data\\raw\\" #path of the files
save_path = "d:\\data\\data_cleaned"
output = "d:\\data\\data_output\\"

#set float format
pd.set_option('display.float_format', lambda x: '%.2f' % x)

filesize = len(listdir(filename)) #get the number of pickled files in the data folder

#load pickled files
for a,b in enumerate(listdir(filename)):#laod the data frames as pandas pd
    vars()['df' + str(a)] = pd.DataFrame()
    vars()['df' + str(a)] = pd.read_pickle(open((filename + '/' + b), 'rb'))
    print('df' + str(a) + ' is loaded' + ' ' +  b)

### delete undesired durations
for a in range(filesize):
  vars()['df' + str(a)]['duration'] =  vars()['df' + str(a)]['d_timestamp'] -  vars()['df' + str(a)]['o_timestamp']
  vars()['df' + str(a)] = vars()['df' + str(a)][vars()['df' + str(a)]['duration'] != vars()['df' + str(a)]['duration'].min()]

#project coordinates columns: origin and destination -- 
#distance&duration&speed calculate
for a in range(filesize):
  transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
  vars()['df' + str(a)]['o_x'], vars()['df' + str(a)]['o_y'] = transformer.transform(vars()['df' + str(a)]['o_lng'].values, vars()['df' + str(a)]['o_lat'].values)
  vars()['df' + str(a)]['d_x'], vars()['df' + str(a)]['d_y'] = transformer.transform(vars()['df' + str(a)]['d_lng'].values, vars()['df' + str(a)]['d_lat'].values)

  #calculate distance,speed,duration
  vars()['df' + str(a)]['distance'] = np.sqrt(pow(vars()['df' + str(a)]['d_x'] - vars()['df' + str(a)]['o_x'],2) + pow(vars()['df' + str(a)]['d_y'] - vars()['df' + str(a)]['o_y'],2))
  vars()['df' + str(a)]['duration'] = vars()['df' + str(a)]['d_timestamp'] - vars()['df' + str(a)]['o_timestamp']
  vars()['df' + str(a)]['speed'] = (vars()['df' + str(a)]['distance']/1000)/(vars()['df' + str(a)]['duration'].dt.seconds/3600)
  vars()['df' + str(a)]['du'] = vars()['df' + str(a)]['duration'].dt.seconds/60
  print('df' + str(a) + 'transformed')

  #remove 0 values
  vars()['df' + str(a)] = vars()['df' + str(a)][vars()['df' + str(a)]['distance'] != 0]
  vars()['df' + str(a)] = vars()['df' + str(a)][vars()['df' + str(a)]['duration'] != vars()['df' + str(a)]['duration'].min()]
  vars()['df' + str(a)] = vars()['df' + str(a)][vars()['df' + str(a)]['speed'] != 0]

#data cleansing + limit data within threshold
for a in range(filesize):
  df = vars()['df' + str(a)]
  vars()['df' + str(a)] = df.loc[(df['distance'] >= 200) & (df['distance'] <= 5000)]
  vars()['df' + str(a)] = df.loc[(df['du'] >= 0.5) & (df['du'] <= 40)]
  vars()['df' + str(a)] = df.loc[(df['speed'] >= 4) & (df['speed'] <= 25)]
  del df
 
    
#CLIP within inner road
innerpath = 'd:\\data\\inner\\inner.shp'
inner = gpd.read_file(innerpath)

for a,b in enumerate(listdir(filename)):
    df = vars()['df' + str(a)]
    #create geo file from origin coords
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.o_lng, df.o_lat),crs="EPSG:4326")
    gdf = gpd.clip(gdf,inner)
    gdf = gpd.GeoDataFrame(
    gdf, geometry=gpd.points_from_xy(gdf.d_lng, gdf.d_lat),crs="EPSG:4326")
    vars()['df' + str(a)] = gpd.clip(gdf,inner)
    print('df' + str(a) + 'clipped')
    

# save files
os.chdir("d:\\data\\data_cleaned\\")
for a in range(filesize):
    df = open('df' + str(a) + '.pkl', 'wb')
    pickle.dump(vars()['df' + str(a)], df)
    df.close()
    print('df' + str(a) + ' is pickled')

##merge file into one named df
frames = []
for x in range(filesize):
  frames.append(vars()['df' + str(x)])
df = pd.concat(frames)

df_sample = df.sample(500000)

#pickle merged trips 
dfpath = open(output + 'trips' +'.pkl', 'wb')
pickle.dump(df, dfpath)
dfpath.close()
print('trips saved')

#pickle sampled trips
df_sample_path = open(output + 'trips_sampled' +'.pkl', 'wb')
pickle.dump(df_sample, df_sample_path)
df_sample_path.close()
print('trips saved')