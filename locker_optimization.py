# -*- coding: utf-8 -*-
# @Author  : Birkan Caliskan
# @Email   : birkanx@gmail.com


import pandas as pd
import numpy as np

dfpath = 'd:\\data\\'
output = "d:\\data\\data_output\\"

#unpickle trips
tripspath = 'd:\\data\\data_output\\trips_sampled.pkl'
df = pd.DataFrame()
df = pd.read_pickle(open(tripspath, 'rb'))

n=25 #number of stations 

sta= pd.read_csv(output + '/' + 'stations.csv')[['x','y']]#station table with xy coordinates

##locker optimization
#attach demand points to candidate station within 500m coverage limit
#df = dfs[['o_x','o_y','d_x','d_y','o_timestamp','d_timestamp']]
#df['sta'] = dist['sta']
#sta = maxsta

#origin station allocation >> Calculate distances of demand points to every candidate station save into dist dataframe
df.dtypes
sta.dtypes
dist = pd.DataFrame()
df.index = list(range(len(df)))

sta['lockercount'] = np.nan
i = 0
for ind,s in enumerate(sta.iloc):
  i +=1
  dist[str(ind)] = np.sqrt(pow(df['o_x'] - s.x ,2) + pow(df['o_y'] - s.y,2))

#destination station allocation >> Calculate distances of demand points to every candidate station save into dist dataframe
d_dist = pd.DataFrame()
i = 0
for ind,s in enumerate(sta.iloc):
  i +=1
  d_dist[str(ind)] = np.sqrt(pow(df['d_x'] - s.x ,2) + pow(df['d_y'] - s.y,2))


(d_dist['11'] < 500).sum() # test the coverage, just data integrity

##remove distances less than interval values for dist and d_dits
for c in range(len(dist.columns)):
  dist[str(c)] = dist[str(c)].mask(dist[str(c)] > 500)

for c in range(len(d_dist.columns)):
  d_dist[str(c)] = d_dist[str(c)].mask(d_dist[str(c)] > 500)

# Create time dataframe for station change the station size
tim = pd.DataFrame(columns=['l'])
for r in range(n):
  tim.loc[r] = np.nan

#allocate origin demand points to stations > find out which station covers an origin demand point
dist['sta'] = np.nan
for x in range(len(dist.columns)-1):
  dist.loc[dist[str(x)] < 500,'sta']
  dist.loc[dist[str(x)] < 500,'sta'] = int(x)

#count locker coverage
for ind,s in enumerate(sta.iloc):
  totaldemand = dist[dist['sta'] == ind].count()[ind]
  sta.loc[ind,'lockercount']  = totaldemand
      
#allocate dest demand points to stations > find out which station covers a destionation demand point
d_dist['sta'] = np.nan
for x in range(len(d_dist.columns)-1):
  d_dist.loc[d_dist[str(x)] < 500,'sta'] = int(x)
#dist['sta'][dist['sta'] == 0] = np.nan
#(dist['sta'] == 0).sum()

#save dist 
#dist.to_csv('/content/drive/MyDrive/coverage/dist' + str(n) + '.csv')
#d_dist.to_csv('/content/drive/MyDrive/coverage/d_dist'+ str(n) + '.csv')

#load dist from drive
#dist = pd.read_csv('/content/drive/MyDrive/coverage/dist.csv', index_col = 0)
#d_dist = pd.read_csv('/content/drive/MyDrive/coverage/d_dist.csv', index_col = 0)

#Distance check between stations
for m in range(len(sta)):
  for z in range(len(sta)):
    test = np.sqrt(pow(sta.iloc[m]['x'] - sta.iloc[z]['x'] ,2) + pow(sta.iloc[m]['y'] - sta.iloc[z]['y'],2))
    print(test)

#group timestamps into 5mins intervals
#origin
df['sta'] = dist['sta']
df['d_sta'] = d_dist['sta']

o_df = df[['o_timestamp','sta']].groupby([df['o_timestamp'].dt.floor('5min'), 'sta']).count()
d_df = df[['d_timestamp','d_sta']].groupby([df['d_timestamp'].dt.floor('5min'), 'd_sta']).count()

#change column name
o_df.columns = ['o_t']
d_df.columns = ['d_t']

#index to column
o_df = o_df.reset_index()
d_df = d_df.reset_index()

#extract timestamps as a column
o_df['o_time']=o_df['o_timestamp'].astype(str).str.split().str[1]
d_df['d_time']=d_df['d_timestamp'].astype(str).str.split().str[1]

#group by 5 mins interval
o_df = o_df[['o_time','sta','o_t']].groupby(['o_time','sta']).sum()
d_df = d_df[['d_time','d_sta','d_t']].groupby(['d_time','d_sta']).sum()

#reset index
o_df = o_df.reset_index()
d_df = d_df.reset_index()

#reshape df
o_df = o_df.pivot(index='sta', columns='o_time', values= 'o_t')
d_df = d_df.pivot(index='d_sta', columns='d_time', values= 'd_t')

#locker optimization
lock = pd.DataFrame(columns=['mi','ma','lock'])
for x in range(len(d_df.T.columns)):
  
  mi = (o_df.T - d_df.T)[x].min()
  ma = (o_df.T - d_df.T)[x].max()
  if abs(mi)>abs(ma):
    lo = abs(mi)
  else:
    lo = abs(ma)
  lock = lock.append({
      'mi':mi,
      'ma':ma,
      'lock':int(lo)
  },ignore_index=True)

lock['lock'] = lock['lock'].astype(int)
sta['lock'] = lock['lock']
sta.to_csv(output + 'stationlockers' + str(n) + '.csv')
