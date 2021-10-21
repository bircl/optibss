# -*- coding: utf-8 -*-
# @Author  : Birkan Caliskan
# @Email   : birkanx@gmail.com

import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import random
from shapely.geometry import MultiLineString
from shapely.ops import polygonize


dfpath = 'd:\\data\\'
output = "d:\\data\\data_output\\"

#unpickle trips
tripspath = 'd:\\data\\data_output\\trips_sampled.pkl'
df = pd.DataFrame()
df = pd.read_pickle(open(tripspath, 'rb'))

#divide study area into grids
dfs= df[['o_x','o_y','d_x','d_y']]
dfs['count'] = 0
gridsize = 50
testg = gpd.GeoDataFrame(geometry = gpd.points_from_xy(dfs['o_x'],dfs['o_y'], crs =32651))
[xmin,ymin,xmax,ymax] = testg.geometry.total_bounds #get spatial bound
gridx = np.arange(xmin,xmax,gridsize) #1891 * 1700 grid to cover study area
gridy = np.arange(ymin,ymax,gridsize) #create list of center coordinate of grids
len(gridx) , len(gridy)

hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(gridx[:-1], gridx[1:]) for yi in gridy]
vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(gridy[:-1], gridy[1:]) for xi in gridx]

grids = list(polygonize(MultiLineString(hlines + vlines)))
id = [i for i in range(len(grids))]
gridgeo = gpd.GeoDataFrame({"id":id,"geometry":grids},crs=32651)

#countup points in polygons

from geopandas.tools import sjoin
pointInPolys = sjoin(testg, gridgeo, how='left')
agg_points = (pointInPolys.groupby(['id']).size().reset_index(name='count')).sort_values('count', ascending=False)

#join df grid and aggregate
gridgeo = gridgeo.join(agg_points.set_index('id'), on='id').dropna()
#calculate centroid of polygons and join
gridgeo['cen'] = gridgeo.centroid

#grid random choose

gridm = gridgeo
#gridm = gridgeo[gridgeo['count'] >= gridgeo['count'].mean()*5]
gridt = pd.DataFrame()
gridt['x'],gridt['y'] = gridm.cen.x, gridm.cen.y
gridt = gridt.reset_index()

#crossover function
def crossover(surv):
  for v in range(int(crossRate*p)):
    rndCross = random.choices(range(p), weights=individual['eval'], k=2) #pick two individual to be mated
    cuttingPoint = random.randint(0,n-1) # cut from randomly choosed edge
    bfrcut = randomPopulation.iloc[rndCross[0]][cuttingPoint:n] # cut chromosomes after the cutting point
    aftcut = randomPopulation.iloc[rndCross[1]][0:cuttingPoint] # cut chromosomes before cutting point
    for z in range(len(bfrcut)):
      for y in range(len(aftcut)):
        while all(bfrcut.iloc[z][['x','y']] == aftcut.iloc[y][['x','y']]) == True:
          gridrandom = gridt.iloc[np.random.choice(gridt.index)]
          aftcut.iloc[y,aftcut.columns.get_loc('x')] = gridrandom.x
          aftcut.iloc[y,aftcut.columns.get_loc('y')] = gridrandom.y
    child = bfrcut.append(aftcut) #concat chromosomes
    child.index = list(range(n))
    children[v] = child #add each offspring into children df
    survivors[survival+v] = child #add survivors to population
    #children = zip(child)
    randomPopulation = survivors

#mutation function  
def mutation(indf):
  global children
  rndind = random.sample(range(p-survival), k=int(mutationrate*p)) #select individuals to be mutated
  for m in rndind:
    rndedge = sorted(random.sample(range(n+1), k=2))
    rndgenesdif = rndedge[1]-rndedge[0]
    gridrandom = gridt.iloc[np.random.choice(gridt.index,rndgenesdif)]
    rndgenes = pd.DataFrame(np.dstack((gridrandom.x, gridrandom.y))[0],columns=['x','y'])
    rndgenes.index = indf.loc[m][rndedge[0]:rndedge[1]].index
    children.loc[m].loc[rndedge[0]:rndedge[1],'x':'y'] = rndgenes
  return rndgenes,rndedge,rndgenesdif,rndind

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#genetic algorithm
n = 25 #candidate station number
p = 100 #population size
gen = 10 #generation number
survival = round(p*0.05)
crossRate = ((p-survival)/p)
mutationrate = 0.1
intervals = 500 #max distance from demand points to station
ind = 0
leftover = int(p-survival-crossRate*survival)
offspring = int(survival+crossRate*survival)

#pandas add candidate location df: x, y, coverage number, ev
#add past candidate check for current randomly selected stations
#define goal situation. we need to evaluate proposed stations by the goal situation. evaluation
goal = len(dfs)
tempcands = pd.DataFrame(columns=['x','y','cover','eval'])
cands = pd.DataFrame(columns=['x','y','cover','eval'])

individual = pd.DataFrame(index=list(range(p)),columns=['cover','eval','dist'])
children = pd.DataFrame(zip([pd.DataFrame({0}),pd.DataFrame({0})]))[0]
coverage = list()
individuals = pd.DataFrame()
testlog = pd.DataFrame()
gridrandom = gridt.iloc[np.random.choice(gridt.index,n)]
randomPopulation = pd.DataFrame(zip([pd.DataFrame(np.dstack((gridrandom.x, gridrandom.y))[0],columns=['x','y']) for x in range(p)]))[0]

for g in range(gen): #iterate generations
   #generate random population size individuals
  '''
  if g != 1:
    for x,l in zip(range(survival),ind): #add survivors to population
      randomPopulation[p+x] = survivors[l]
  '''

  #rnd = np.dstack((np.random.choice(gridx,n), np.random.choice(gridy,n))[0]#choose n random stations in 50*50m grid
  #rnd = np.dstack(rnd)[0] #stack n station
  total = dfs
  for s,d in enumerate(randomPopulation): #iterate calculating distance from candidates
    d['cover'],d['eval'],d['dist'] = np.nan, np.nan, np.nan #initiate columns
    for j,h in enumerate(d.iloc): #choose each gene in individual
      tempdist = np.sqrt(pow(dfs['o_x'] - h.x ,2) + pow(dfs['o_y'] - h.y,2)) #distance
      cover = len(dfs.loc[tempdist < intervals]) #filter less than intervals meter and sum
      #print(d[0],d[1],cover)
      d.loc[j]['cover'] = cover
      d.loc[j]['dist'] = tempdist.sum()
      
    individual.iloc[s,individual.columns.get_loc('cover')] = d['cover'].sum() #sum up covered stations indiviuals
  individual['eval'] = individual['cover']/individual['cover'].sum() #evaluate individuals
  coverage.append(individual['cover'].max())
  #ind = np.random.choice(list(individual.index),size=survival, p=individual['eval'], replace=False) #roulette wheel choose from the population
  individual['eval'] = individual['eval'].astype(float)
  ind = individual.nlargest(survival, ['eval']).index
  survivors = randomPopulation[ind] #survivors of the population
  survivors.index = list(range(survival)) # reset index of the survivors
  testlog[str(g)] = individual['cover']
  #crossover(randomPopulation) #crossover among survivors
  #print(survivors[0])
  ###crossover'''
  for v in range(int(crossRate*p)):
    rndCross = random.choices(range(p), weights=individual['eval'], k=2) #pick two individual to be mated
    cuttingPoint = random.randint(0,n-1) # cut from randomly choosed edge
    bfrcut = randomPopulation.iloc[rndCross[0]][0:cuttingPoint] # cut chromosomes after the cutting point
    aftcut = randomPopulation.iloc[rndCross[1]][cuttingPoint:n] # cut chromosomes before cutting point
    #'''
    for z in range(len(bfrcut)):
      for y in range(len(aftcut)):
        if bfrcut.iloc[z]['cover'] == 0:
          gridrandom = gridt.iloc[np.random.choice(gridt.index)]
          bfrcut.iloc[z,bfrcut.columns.get_loc('x')] = gridrandom.x
          bfrcut.iloc[z,bfrcut.columns.get_loc('y')] = gridrandom.y
        if aftcut.iloc[y]['cover'] == 0:
          gridrandom = gridt.iloc[np.random.choice(gridt.index)]
          aftcut.iloc[y,aftcut.columns.get_loc('x')] = gridrandom.x
          aftcut.iloc[y,aftcut.columns.get_loc('y')] = gridrandom.y
        while bfrcut.iloc[z]['x'] == aftcut.iloc[y]['x'] and bfrcut.iloc[z]['y'] == aftcut.iloc[y]['y']:
          gridrandom = gridt.iloc[np.random.choice(gridt.index)]
          aftcut.iloc[y,aftcut.columns.get_loc('x')] = gridrandom.x
          aftcut.iloc[y,aftcut.columns.get_loc('y')] = gridrandom.y
    #'''
    child = bfrcut.append(aftcut) #concat chromosomes
    child.index = list(range(n))
    children[v] = child #add each offspring into children df
    survivors[survival+v] = 0
    #survivors[survival+v] = child #add survivors to population
    
    #children = zip(child)
  mutation(children)  
  survivors[survival:p] = children #add crossovered individuals to population
  randomPopulation = survivors

  from sklearn.metrics.pairwise import nan_euclidean_distances
  from scipy.spatial.distance import cdist
  #distance check between stations
  for po in range(100):
    sta_coordinates = randomPopulation[po][['x','y']]
    dista = nan_euclidean_distances(sta_coordinates, sta_coordinates) < 500
    for di in range(n):
      #print(dista[di])
      for bi in range(n):
        if bi != di and dista[di][bi] == True:
          gridrandom = gridt.iloc[np.random.choice(gridt.index)]
          randomPopulation[po].iloc[bi].x = gridrandom.x
          randomPopulation[po].iloc[bi].y = gridrandom.y

  #randomPopulation = randomPopulation.sample(frac=1) #shuffle population dataframe
  randomPopulation.index = list(range(p)) #rearrange index of the population
  #randomPopulation.to_csv('/content/drive/MyDrive/coverage/randomPopulationtest.csv')
  print(coverage[g])
  #d['eval'] = d['cover'].sum/d()['cover'].sum() # fill evaluation
  #ind = tempcands.iloc[np.random.choice(list(tempcands.iloc[-n:].index),size=survival, p=tempcands['eval'].iloc[-n:], replace=False)] #np.random.choice(list(tempcands.iloc[-5:].index),size=2, p=tempcands['eval'], replace=False)
  #cands = cands.append(roulette)

  #save
  randompoppath = open(output + 'randompop.pkl', 'wb')
  pickle.dump(randomPopulation,randompoppath)
  randompoppath.close()
  #save parameters
  pd.DataFrame({'gen':[g+1]}).to_csv(output + 'param.csv')


pd.DataFrame(coverage).to_csv(output + 'coverages' + str(n) + 'g' +str(gen) + 'c' +str(crossRate) + 'm' +str(mutationrate) + 'p' + str(p) + '.csv')
len(randomPopulation)
totalcov = pd.DataFrame()
for en,x in enumerate(randomPopulation):
  totalcov = totalcov.append({'cov': x.cover.sum(),
                             'index':en
  },ignore_index=True)
maxidcov = totalcov['cov'].idxmax()
randomPopulation.loc[maxidcov].to_csv(output + 'stations' + str(n) + 'g' +str(gen) + 'c' +str(crossRate) + 'm' +str(mutationrate) + 'p' +str(p) + '.csv')
