#coding=utf-8  / #coding=gbk 


#如果跑出的数据有问题，看看是不是画图的程序有问题

import pyproj as proj4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import scipy.io
#import read_logs
#from mpl_toolkits.basemap import Basemap
from coordinateSystems import TangentPlaneCartesianSystem, GeographicSystem, MapProjection
from scipy.stats import norm
import cmaps
import operator
from functools import reduce
import simulation_functions as sf
import numpy as np
from scipy.linalg import lstsq# Return the least-squares solution to a linear matrix equation
from scipy.optimize import leastsq#
from coordinateSystems import GeographicSystem
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

    

c0 = 3.0e8 # m/s
dt_rms = 100.e-9 # seconds
lma_digitizer_window = 40.0e-9 # seconds per sample ##需要改吗20

#dbw=10*lg(P/1w)
#dbm=10*lg(P/1mw)
#1w=0dbw=30dbm
#dbm=30+dbw


# ### Station coordinates from csv file
fname1= r'D:\zhanghuiyi-2021-2024\02_python\demo_montecarlo.csv'#sys.argv[1]
# Input network title and csv file here

Network = 'test' # name of network in the csv file
stations = pd.read_csv( fname1) # network csv file with one or multiple networks

stations.set_index('network').loc[Network]
aves = np.array(stations.set_index('network').loc[Network])[:,:-1].astype('float')

# ### Setting up and checking station locations
center = (np.mean(aves[:,1]), np.mean(aves[:,2]), np.mean(aves[:,0])) 
geo  = GeographicSystem()
tanp = TangentPlaneCartesianSystem(center[0], center[1], center[2])
mapp = MapProjection
projl = MapProjection(projection='laea', lat_0=center[0], lon_0=center[1])

alt, lat, lon  = aves[:,:3].T
stations_ecef  = np.array(geo.toECEF(lon, lat, alt)).T
stations_local = tanp.toLocal(stations_ecef.T).T

center_ecef = np.array(geo.toECEF(center[1],center[0],center[2]))
ordered_threshs = aves[:,-1]

plt.scatter(stations_local[:,0]/1000., stations_local[:,1]/1000., c=aves[:,3])
plt.colorbar()
############## 38732.09024408967;   28186.978934259496
circle=plt.Circle((0,0),30,color='k',fill=False)
# plt.xlim(-80,80)
# plt.ylim(-80,80)
# fig = plt.gcf()
# fig.gca().add_artist(circle)
plt.show()


# np.max(stations_local[:,1])-np.min(stations_local[:,1])

# ### Setting up grid
# 
# Input desired grid boundaries and interval here in meters from the center of the network (no point located over the center!)


xmin, xmax, xint = -200001, 199999,5000
ymin, ymax, yint = -200001, 199999, 5000
# alts = np.arange(2000,20001,2000.)
alts = np.array([7000])

initial_points = np.array(np.meshgrid(np.arange(xmin,xmax+xint,xint),
                                      np.arange(ymin,ymax+yint,yint), alts))  #xyz*81*81*1

#a=initial_points[1,:,:,0]


x,y,z=initial_points.reshape((3,np.size(initial_points)//3))  #转一维  #xyz*6561
points2 = tanp.toLocal(np.array(projl.toECEF(x,y,z))).T

means = np.empty(np.shape(points2))
stds  = np.empty(np.shape(points2))
misses= np.empty(np.shape(points2))
rmses= np.empty(np.shape(points2))
chi22 = np.empty(np.shape(points2[:,0]))
rmses_r= np.empty(np.shape(points2))
rmses_r_chi2_1= np.empty(np.shape(points2))



tanp_all = []
for i in range(len(aves[:,0])): 
    tanp_all = tanp_all + [TangentPlaneCartesianSystem(aves[i,1],aves[i,2],aves[i,0])]


# ### General calculations at grid points
# Set number of iterations and solution requirements here

iterations=50  #迭代100次？
chi22_all=[];
# # for r,theta,z errors and standard deviations and overall detection efficiency
for i in range(len(x)):#对每个子网格进行模拟
    means[i], stds[i], misses[i] ,rmses[i],chi22 ,rmses_r[i],rmses_r_chi2_1[i]= sf.black_box(points2[i,0], points2[i,1], points2[i,2], 
             iterations,
             stations_local,
             ordered_threshs,
             stations_ecef,center_ecef,tanp_all, c0,dt_rms,tanp,projl,
             chi2_filter=5.,
             min_stations=4,
             just_rms=False
             )
    chi22_all.append(chi22)
# Just rmse values:
# for i in range(len(x)):
#     means[i] = sf.black_box(x[i], y[i], z[i], iterations,
#               stations_local,ordered_threshs,stations_ecef,center_ecef,
#               tanp_all,c0,dt_rms,tanp,projl,
#               chi2_filter=5.,min_stations=6,just_rms=True
#               )

means  = (means.T.reshape(np.shape(initial_points)))
stds   = (stds.T.reshape(np.shape(initial_points)))
misses = (misses.T.reshape(np.shape(initial_points)))
rmses = (rmses.T.reshape(np.shape(initial_points)))
rmses_r = (rmses_r.T.reshape(np.shape(initial_points)))
rmses_r_chi2_1= (rmses_r.T.reshape(np.shape(initial_points)))
means  = np.ma.masked_where(np.isnan(means) , means)# 对nan值进行掩码
stds   = np.ma.masked_where(np.isnan(stds)  , stds)
misses = np.ma.masked_where(np.isnan(misses), misses)
rmses = np.ma.masked_where(np.isnan(rmses), rmses)
rmses_r = np.ma.masked_where(np.isnan(rmses_r), rmses_r)
rmses_r_chi2_1 = np.ma.masked_where(np.isnan(rmses_r_chi2_1), rmses_r_chi2_1)
# mtest=means[:,0].reshape(81,81)




chi2_array=np.array(chi22_all)
chi2_array2  = np.array([i for item in chi2_array for i in item ])
chi2_array3  =np.array( [x for x in chi2_array2 if x ==x])
# np.savetxt(r'E:\important_copy\code\小论文\out\chi2_0606.csv', chi2_array3,delimiter=' ')


# 获取当前日期
current_date = datetime.now().strftime('%Y%m%d')

# 生成带日期的文件名
filename = f'D:\\zhanghuiyi-2021-2024\\01_sta_network_evaluation\\1-4绘图\\test_{current_date}.mat'

# 保存数据到MAT文件
scipy.io.savemat(filename, {
    'means': means,
    'misses': misses,
    'stds': stds,
    'rmses': rmses,
    'rmses_r': rmses_r,
    'rmses_r_chi2_1': rmses_r_chi2_1,
    'stations_local': stations_local
})
