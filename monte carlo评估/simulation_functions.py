######################################################
# Various functions for LMA simulations
#
# The black_box function is the main driver function
# 
# Contact:
# vanna.chmielewski@ttu.edu
#
#
# This model and its results have been submitted to the Journal of Geophysical Research.
# Please cite:
# V. C. Chmielewski and E. C. Bruning (2016), Lightning Mapping Array flash detection 
# performance with variable receiver thresholds, J. Geophys. Res. Atmos., 121, 8600-8614, 
# doi:10.1002/2016JD025159
#
# If any results from this model are presented.
######################################################

import numpy as np
from scipy.linalg import lstsq#最小二乘法 Return the least-squares solution to a linear matrix equation
from scipy.optimize import leastsq#最小二乘拟合
from coordinateSystems import GeographicSystem
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def travel_time(X, X_ctr, c, t0=0.0, get_r=False):  #################station locations,
    """ Units are meters, seconds.
        X is a (N,3) vector of source locations, and X_ctr is (N,3) receiver
        locations.

        t0 may be used to specify a fixed amount of additonal time to be
        added to the travel time.

        If get_r is True, return the travel time *and* calculated range.
        Otherwise
        just return the travel time.
    """
    ranges = (np.sum((X[:, np.newaxis] - X_ctr)**2, axis=2)**0.5).T#np.newaxis增加新维度
    time = t0+ranges/c
    if get_r == True:
        return time, ranges
    if get_r == False:
        return time


#频率越高，波长越短，功率衰减越多，，，，混频的影响有多大，再考虑位置，它的贡献小是因为布局还是频段
#把193的全部换为63的探测范围肯定变大
def received_power(power_emit, r, wavelength=1000, recv_gain=1.0): #3e8/3e5=1000
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave. #3e8/63e6=4.762 #波长=速度（光速）/频率
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r  # r是travel_time里的ranges
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)# P_received=P_emitted(wavelength/4_pi_r)^2
    return recv_gain * power_emit * free_space_loss
def received_power1(power_emit, r, wavelength=1.554, recv_gain=1.0):    #频段1前八站########193 MHz
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 193 MHz wave. #3e8/193e6=1.554 #波长=速度（光速）/频率
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r  # r是travel_time里的ranges
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)# P_received=P_emitted(wavelength/4_pi_r)^2
    return recv_gain * power_emit * free_space_loss
def received_power2(power_emit, r, wavelength=4.762, recv_gain=1.0):  #频段2后三站#######63 MHz
    """ Calculate the free space loss. Defaults are for no receive gain and
        a 63 MHz wave. #3e8/63e6=4.762
        Units: watts, meters
    """
    four_pi_r = 4.0 * np.pi * r  # r是travel_time里的ranges
    free_space_loss = (wavelength*wavelength) / (four_pi_r * four_pi_r)# P_received=P_emitted(wavelength/4_pi_r)^2
    return recv_gain * power_emit * free_space_loss






def precalc_station_terms(stations_ECEF):######## station locations
    """ Given a (N_stn, 3) array of station locations,
        return dxvec and drsq which are (N, N, 3) and (N, N),
        respectively, where the first dimension is the solution
        for a different master station
    """
    dxvec = stations_ECEF-stations_ECEF[:, np.newaxis]
    r1    = (np.sum(stations_ECEF**2, axis=1))
    drsq  = r1-r1[:, np.newaxis]
    return dxvec, drsq


def linear_first_guess(t_i, dxvec, drsq, c=3.0e8):#线性解
    """ Given a vector of (N_stn) arrival times,
        calcluate a first-guess solution for the source location.
        Return f=(x, y, z, t), the source retrieval locations.
    """
    g = 0.5*(drsq - (c**2)*(t_i**2))
    K = np.vstack((dxvec.T, -c*t_i)).T
    f, residuals, rank, singular = lstsq(K, g)
    return f


def predict(p, stations_ECEF2, c=3.0e8):#迭代方程
    """ Predict arrival times for a particular fit of x,y,z,t source
        locations. p are the parameters to retrieve
    """
    #p=p0
    #c=c0
    return (p[3] + ((np.sum((p[:3] - stations_ECEF2) *
            (p[:3] - stations_ECEF2), axis=1))**0.5)) / c


def residuals(p, t_i, stations_ECEF2):#误差函数
    return t_i - predict(p, stations_ECEF2)


def dfunc(p, t_i, stations_ECEF2, c=3.0e8):
    return -np.vstack(((p[:3] - stations_ECEF2).T / (c *
              (np.sum((p[:3] - stations_ECEF2) *
              (p[:3] - stations_ECEF2), axis=1))**0.5),
             np.array([1./c]*np.shape(stations_ECEF2)[0])))


def gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq, center_ECEF,
                       stations_ECEF, dt_rms, min_stations=5,
                       max_z_guess=25.0e3):
    """ t_all is a N_stations x N_points masked array of arrival times at
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal

        center_ECEF for the altitude check

        This streamlines the generator function, which emits a stream of 
        nonlinear least-squares solutions.
    """  
  
    m = t_mins[i]#对应的第几个站点（具有最小到达时间，最先被检索到的）
    stations_ECEF2=stations_ECEF[selection]#selection是被哪些站接收，每个selection<=站数，坐标转换（三维）
    # Make a linear first guess
    #得到第一次猜测结果p0（x,y,z,t）
    p0 = linear_first_guess(np.array(t_all[:,i][selection]-t_all[m,i]),
                            dxvec[m][selection], 
                            drsq[m][selection])
    t_i =t_all[:,i][selection]-t_all[m,i]#t_i是以最小到达时间为起点，得到每个站与它的时间差
    # Checking altitude in lat/lon/alt from local coordinates
    latlon = np.array(GeographicSystem().fromECEF(p0[0], p0[1],p0[2]))
    if (latlon[2]<0) | (latlon[2]>20000): #不在高度范围内则高度取7000m
        latlon[2] = 7000
        new = GeographicSystem().toECEF(latlon[0], latlon[1], latlon[2])
        p0[:3]=np.array(new)
        
        
    plsq = np.array([np.nan]*5)   
    
    #####最大迭代次数是否设限？20？
    plsq[:4], cov, infodict, mesg,ier = leastsq(residuals, p0, #residuals，计算方程，p0,初始时间，
                            args=(t_i, stations_ECEF2),        #args，其他参数，（到达时间，站点位置）（参与在residuals中）
                            Dfun=dfunc,col_deriv=1,full_output=True) 
    ####卡方
    plsq[4] = np.sum(infodict['fvec']*infodict['fvec'])/(      #'fvec'协方差矩阵？
                     dt_rms*dt_rms*(float(np.shape(stations_ECEF2)[0]-4)))#dt_rms*dt_rms时间方差
    return plsq

def gen_retrieval(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF, 
                  dt_rms, min_stations=4, max_z_guess=25.0e3): 
    """ t_all is a N_stations x N_points masked array of arrival times at 
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal
    
        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.
    """    
    #t_mins=dt_mins
    #dxvec=full_dxvec
    #drsq=full_drsq
    #center_ECEF=center_ecef
    #stations_ECEF=stations_ecef
    #min_stations=6
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmask(t_all[:,i])#没有被mask掉的值为true，每一列（每一个发射源）
        if np.all(selection == True):#如果全部都为true，即一个点被所有站接收到
            selection = np.array([True]*len(t_all[:,i]))
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms, 
                                     min_stations, max_z_guess=25.0e3)
        elif np.sum(selection)>=min_stations:#如果大于最少站数
            yield gen_retrieval_math(i, selection, t_all, t_mins, dxvec, drsq,
                                     center_ECEF, stations_ECEF, dt_rms, 
                                     min_stations, max_z_guess=25.0e3)
        else: #其他情况返回无效值
            yield np.array([np.nan]*5)

def gen_retrieval_full(t_all, t_mins, dxvec, drsq, center_ECEF, stations_ECEF, 
                       dt_rms, c0, min_stations=6, max_z_guess=25.0e3): 
    """ t_all is a N_stations x N_points masked array of arrival times at 
        each station.
        t_min is an N-point array of the index of the first unmasked station 
        to receive a signal
    
        center_ECEF for the altitude check

        This is a generator function, which emits a stream of nonlinear
        least-squares solutions.

        Timing comes out of least-squares function as t*c from the initial 
        station
    """    
    for i in range(t_all.shape[1]):
        selection=~np.ma.getmask(t_all[:,i])
        plsq = np.array([np.nan]*7)
        if np.all(selection == True):
            selection = np.array([True]*len(t_all[:,i]))
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                         drsq, center_ECEF, stations_ECEF, dt_rms, 
                         min_stations, max_z_guess=25.0e3)
            
            
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            
            
            yield plsq
        elif np.sum(selection)>=min_stations:
            plsq[:5] = gen_retrieval_math(i, selection, t_all, t_mins, dxvec,
                                     drsq, center_ECEF, stations_ECEF, dt_rms,
                                     min_stations, max_z_guess=25.0e3)
            plsq[5] = plsq[3]/c0 + t_all[t_mins[i],i]
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq
        else: 
            plsq[6] = np.shape(stations_ECEF[selection])[0]
            yield plsq


def black_box(x,y,z,n,
              stations_local,ordered_threshs,stations_ecef,center_ecef,
              tanps,
              c0,dt_rms,tanp,projl,chi2_filter,min_stations=4,just_rms=False):
    """ This funtion incorporates most of the Monte Carlo functions and calls
        into one big block of code.

        x,y,z are the source location in the local tangent plane (m)
        n is the number of iterations

        stations_local is the the (N-stations, 3) array of station locations
        in the local tangent plane

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        stations_ecef is (N,3) array in ECEF coordinates

        center_ecef is just the center location of the network, just easier
        to pass into the fuction separately to save some calculation

        c0 is th speed of light

        dt_rms is the standard deviation of the timing error (Gaussian, in s)

        tanp is the tangent plane object

        projl is the map projection object

        chi2_filter is the maximum allowed reduced chi2 filter for the
        calculation (use at most 5)

        min_stations is the minimum number of stations required to receive
        a source. This must be at least 5, can be higher to filter out more
        poor solutions

        If just_rms is True then only the rms error will be returned in a 
        (x,y,z) array.
        If false then mean error (r,z,theta), standard deviation (r,z,theta)
        both in units of (m,m,degrees)
        and the number of bad/missed solutions will be returned.
    """
    #测试用的
    #n=100
    #i=100
    #x=points2[i,0]
    #y=points2[i,1]
    #z=points2[:,2]
    
    
    
    points = np.array([np.zeros(n)+x, np.zeros(n)+y, np.zeros(n)+z]).T#单个格点进行多少次迭代（发射多少个点）点，单个点坐标*n，二维数组
    powers = np.empty(n)#随机赋值，n个一样的值，一维数组
    
    # # For the old 1/p distribution:
    # powers = np.random.power(2, size=len(points[:,0]))**-2
    '''
    #可以试试提高功率，后面三站基本达不到阈值，，，，，，如何确定功率发射函数
    ##63的接收阈值会更高一点（接收范围更大），，，跟底噪相关，，阈值高低的影响，，
    '''
    # For high powered sources (all stations contributing):
    powers[:] = 100000
  
    # For the theoretical distribution:
    # for i in range(len(powers)):#源发射功率的随机选择，在1000w内
    # #为每个发射点选择功率，得到n个不一样的发射功率，即得到n个坐标一样但功率不一样的点
    #     powers[i] = np.max(1./np.random.uniform(0,1000,2000))#np.random.uniform(low, high ,size) 
    #     #####-30--40dbw

   # pwr1 = received_power1(powers, 100000)
   # a= 10.*np.log10(pwr1/1e-3)
   # np.mean(powers)
   # np.min(a)
   
    # Calculate distance and power retrieved at each station and mask
    # the stations which have higher thresholds than the retrieved power

    points_f_ecef = (tanp.fromLocal(points.T)).T  #发射点的坐标系转换，单个点坐标*n
    
    #---------------------------------------------------------------------------分段
    #-----------------------------------------------------------------------------
    
    # #n个一样的点，给出到达不同站点的时间差和距离
    # #points模拟点的起始位置，station_local站点位置，c0为光速，t0=0,get_r=True则返回到达时间差和距离，这个为真实解
    # #1个站得到n个相同的dt和ran
    # dt1, ran1  = travel_time(points, stations_local[0:9,:], c0, get_r=True)#频段1的站
    # dt2, ran2  = travel_time(points, stations_local[9:,:], c0, get_r=True)# 频段2的站
    # dt=np.vstack((dt1,dt2))
    # ran=np.vstack((ran1,ran2))
    # # #考虑功率的距离衰减##每个站对应n个功率不一样的辐射源，后面masking判断每个站的n个点里有多少个是达到阈值而能被接收到的
    # pwr1 = received_power1(powers, ran1)
    # pwr2 = received_power2(powers, ran2)
    # pwr=np.vstack((pwr1,pwr2))
    
    #---------------------------------------------------------------------------不分段
    #--------------------------------------------------------------------------------
    
    dt, ran  = travel_time(points, stations_local, c0, get_r=True)#计算到达每个站的时间差和范围
    pwr = received_power(powers, ran)
#----------------------------------------------------------------------------------
    
#判断每个站有多少个点能被接收到，小于阈值接收不到的则进行掩膜
#masking为N个站*n个点
    masking= 10.*np.log10(pwr/1e-3) < ordered_threshs[:,np.newaxis]  ##dbm=10*lg(P/1mw)
    masking2 = np.empty_like(masking)#np.empty_like生成和已有数组相同大小，类型的数组
    #对于这些点，除了阈值再看看高度是否在海平面以上
    #tanps=tanp_all
    for i in range(len(stations_ecef[:,0])):
        masking2[i]=tanps[i].toLocal(points_f_ecef.T)[2]<0##高度小于0的不要
#选取满足这两个条件的点
    masking = masking | masking2#masking | masking2 or masking2 左边为真取左边值，否则取右边值
    pwr = np.ma.masked_where(masking, pwr)#masking为true的地方对pwr进行掩码
    dt  = np.ma.masked_where(masking, dt)#每个站得到n个一样的值（每一行的值都是一样的）
    ran = np.ma.masked_where(masking, ran)
    
    # Add error to the retreived times 给到达时间增加随机时间差，用于模拟
    '''时间误差或许也可以改一下
    #dt_rms为30ns
    '''
    #此时每个点的到每个站到达时间都是不一样的，（添加误差之前到达同一个站的时间是一样的）
    dt_e = dt + np.random.normal(scale=dt_rms, size=np.shape(dt))#返回一组符合高斯分布的概率密度随机数
    #每一个点找到最小到达时间，对应某一个站，得到n*1个结果
    dt_mins = np.argmin(dt_e, axis=0)#给出axis方向最小值的下标,axis=0每一列
    
    
    # Precalculate some terms in ecef (fastest calculation)
    points_f_ecef = (tanp.fromLocal(points.T)).T  #又换坐标系
    full_dxvec, full_drsq = precalc_station_terms(stations_ecef)#没懂
    # Run the retrieved locations calculation
    # gen_retrieval returns a tuple of four positions, x,y,z,t.
    dtype=[('x', float), ('y', float), ('z', float), ('t', float), 
            ('chi2', float)]
    # Prime the generator function - pauses at the first yield statement.
    point_gen = gen_retrieval(dt_e, dt_mins, full_dxvec, full_drsq, 
                              center_ecef, stations_ecef, dt_rms, 
                              min_stations=4)
    # Suck up the values produced by the generator, produce named array.
    point_gen =list(point_gen)#得到n*5的二维数据，每个有效点的值都不一样
    retrieved_locations= np.vstack(point_gen)
    #retrieved_locations = np.fromiter(point_gen, dtype=dtype)
    retrieved_locations = np.array([(a,b,c,e) for (a,b,c,d,e) in ##
                                     retrieved_locations])
    chi2                = retrieved_locations[:,3]#取chi2列
    retrieved_locations = retrieved_locations[:,:3]##取xyz
    retrieved_locations = np.ma.masked_invalid(retrieved_locations)#无效值进行掩码
    if just_rms == False:
        # Converts to projection
        soluts = tanp.toLocal(retrieved_locations.T)
        # good   = soluts[2] > 0
        proj_soluts = projl.fromECEF(retrieved_locations[:,0], #对有效值再进行坐标转换
                                     retrieved_locations[:,1], 
                                     retrieved_locations[:,2])
        good = proj_soluts[2] > 0                              #解的结果要高度大于0
        proj_soluts = (proj_soluts[0][good],proj_soluts[1][good],#模拟解，不一样的值
                       proj_soluts[2][good])
        proj_points = projl.fromECEF(points_f_ecef[good,0],      #格点值，都是一样的值
                                     points_f_ecef[good,1], 
                                     points_f_ecef[good,2])

        proj_soluts = np.ma.masked_invalid(proj_soluts)
        
        proj_diff=np.array([((proj_points[0]-proj_soluts[0])**2+(proj_points[1]-proj_soluts[1])**2)**0.5,
                            np.degrees(np.arctan(proj_points[0]/proj_points[1]))-np.degrees(np.arctan(proj_soluts[0]/proj_soluts[1])),
                            proj_points[2]-proj_soluts[2]])   
        proj_diff[1][proj_diff[1]>150]=proj_diff[1][proj_diff[1]>150]-180#######？？？这步什么意思
        proj_diff[1][proj_diff[1]<-150]=proj_diff[1][proj_diff[1]<-150]+18
        # Converts to cylindrical coordinates since most errors 
        # are in r and z, not theta 
        proj_points_cyl = np.array([(proj_points[0]**2+proj_points[1]**2)**0.5,  #转为柱坐标
                              np.degrees(np.arctan(proj_points[0]/proj_points[1])),
                              proj_points[2]])
        proj_soluts_cyl = np.ma.masked_array([(proj_soluts[1]**2+proj_soluts[0]**2)**0.5,
                              np.degrees(np.arctan(proj_soluts[0]/proj_soluts[1])),
                              proj_soluts[2]])
        difs = proj_soluts_cyl - proj_points_cyl     #得到差值
        difs[1][difs[1]>150]=difs[1][difs[1]>150]-180#######？？？这步什么意思
        difs[1][difs[1]<-150]=difs[1][difs[1]<-150]+180
        
        difs2   = soluts[:,good] - points[good].T
        return np.mean(difs.T[chi2[good]<chi2_filter].T, axis=1#横向求平均，得到三个数r,θ，z，n个解与真实解的平均误差
             ), np.std(difs.T[chi2[good]<chi2_filter].T, axis=1#axis=1,每一列
             ), np.ma.count_masked(difs[0])+ np.sum(chi2[good]>=chi2_filter)+np.sum(~good##the number of bad/missed solutions
             ),np.mean((difs2.T[chi2[good]<chi2_filter].T)**2, axis=1)**0.5 ,chi2 , np.std(proj_diff.T[chi2[good]<chi2_filter].T, axis=1
             ),np.std(proj_diff.T[chi2[good]<1].T, axis=1)
                                                                                    ##rmes           

    else:
        #Convert back to local tangent plane
        soluts = tanp.toLocal(retrieved_locations.T)
        proj_soluts = projl.fromECEF(retrieved_locations[:,0], 
                                     retrieved_locations[:,1], 
                                     retrieved_locations[:,2])
        good = proj_soluts[2] > 0
        # good   = soluts[2] > 0
        difs   = soluts[:,good] - points[good].T
        return np.mean((difs.T[chi2[good]<chi2_filter].T)**2, axis=1)**0.5
    

def black_box_full(x,y,z,n,
              stations_local,ordered_threshs,stations_ecef,center_ecef,
              c0,dt_rms,tanp,projl,chi2_filter,min_stations=4):
    """ This funtion incorporates most of the Monte Carlo functions and calls
        into one big block of code with other non-standard output. Otherwise
        performs the same basic fuctions as black_box.

        x,y,z are the source location in the local tangent plane (m)
        n is the number of iterations

        stations_local is the (N-stations, 3) array of station locations
        in the local tangent plane

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        stations_ecef is (N,3) array in ECEF coordinates

        center_ecef is just the center location of the network, just easier
        to pass into the fuction separately to save some calculation

        c0 is th speed of light

        dt_rms is the standard deviation of the timing error (Gaussian, in s)

        tanp is the tangent plane object

        projl is the map projection object

        chi2_filter is the maximum allowed reduced chi2 filter for the
        calculation (use at most 5)

        min_stations is the minimum number of stations required to receive
        a source. This must be at least 5, can be higher to filter out more
        poor solutions

        If just_rms is True then only the rms error will be returned in a 
        (x,y,z) array (in m).
        If false then mean error (r,z,theta), standard deviation (r,z,theta)
        both in units of (m,m,degrees) and the number of bad/missed solutions 
        will be returned.
    """
    # Creates an array of poins at x,y,z of n elements
    points = np.array([np.zeros(n)+x, np.zeros(n)+y, np.zeros(n)+z]).T
    powers = np.empty(n)
    for i in range(len(powers)):
        powers[i] = np.max(1./np.random.uniform(0,1000,2000))
    # Calculates distance and powers retrieved at each station to threshold
    dt, ran  = travel_time(points, stations_local, c0, get_r=True)
    pwr = received_power1(powers, ran)
    masking = 10.*np.log10(pwr/1e-3) < ordered_threshs[:,np.newaxis]
    pwr = np.ma.masked_where(masking, pwr)
    dt  = np.ma.masked_where(masking, dt)
    ran = np.ma.masked_where(masking, ran)
    # Adds error to the retreived times
    dt_e = dt + np.random.normal(scale=dt_rms, size=np.shape(dt))
    dt_mins = np.argmin(dt_e, axis=0)
    # Precalculate some terms
    points_f_ecef = (tanp.fromLocal(points.T)).T  
    full_dxvec, full_drsq = precalc_station_terms(stations_ecef)
    # Run the retrieved locations calculation
    # gen_retrieval returns a tuple of four positions, x,y,z,t.
    dtype=[('x', float), ('y', float), ('z', float), 
           ('t', float), ('chi2', float), ('terror',float), 
           ('stations', float)]
    # Prime the generator function - it pauses at the first yield statement.
    point_gen = gen_retrieval_full(dt_e, dt_mins, full_dxvec, full_drsq, 
                                   center_ecef, stations_ecef, dt_rms, 
                                   min_stations) 
    point_gen =list(point_gen)
    retrieved_locations= np.vstack(point_gen)
    # Suck up all the values produced by the generator, produce named array.
   # retrieved_locations = np.fromiter(point_gen, dtype=dtype)
    retrieved_locations = np.array([(a,b,c,e,f,g) for (a,b,c,d,e,f,g) in 
                                   retrieved_locations])
    station_count       = retrieved_locations[:,5]
    terror              = retrieved_locations[:,4]
    chi2                = retrieved_locations[:,3]
    retrieved_locations = retrieved_locations[:,:3]
    retrieved_locations = np.ma.masked_invalid(retrieved_locations)
    # Converts to projection
    soluts = tanp.toLocal(retrieved_locations.T)
    good   = soluts[2] > 0
    station_count[-good] = np.nan
    # proj_points = projl.fromECEF(points_f_ecef[good,0], 
    #                              points_f_ecef[good,1], 
    #                              points_f_ecef[good,2])
    # proj_soluts = projl.fromECEF(retrieved_locations[good,0], 
    #                              retrieved_locations[good,1], 
    #                              retrieved_locations[good,2])
    # proj_soluts = np.ma.masked_invalid(proj_soluts)
    # ranges = (np.sum((retrieved_locations[good,:3][:,np.newaxis] - 
    #                   stations_ecef)**2, axis=2)**0.5).T
    # cal_pwr = (np.mean(pwr*(4*np.pi*ranges/4.762)**2, axis=0))
    # return proj_points, proj_soluts, chi2[good], terror[good], station_count[good]
    # return powers[good], cal_pwr, pwr[0][good]
    # return station_count

def curvature_matrix(points,stations_local,c0,
                     timing_error,min_stations=4):
    """ Calculates the curvature matrix error solutions given stations and
        grid points of sources. Returns array of (x,y,z,ct) errors (in m)

        points is the n by 3 array of source points in x,y,z local tangent
        plane coordinates (m)

        stations_local is the (N-stations, 3) array of station locations
        in the local tangent plane.

        ordered_threshs is the N-station array of thresholds in the same order
        as the station arrays (in dBm).

        c0 is the speed of light

        power is the source power for all of the points in Watts

        timing_error is the Gaussian standard deviation of timing errors of 
        the stations in seconds
    """
    # Find received powers to find which stations contibute for each source
    dist = np.sum((points - stations_local)**2,axis=1)**0.5
    # Precalculate some terms to simplify the calculations for each term
    ut1 = -dist
    ut2 = stations_local - points
    dist_term = dist*dist
    # Find each term of the curvature matrix
    
    curvature_matrix = np.empty((4,4))
    curvature_matrix[0,0] = np.sum((ut2[:,1]**2 + ut2[:,2]**2)*
                                    dist_term**(-3./2.)*ut1+1)
    curvature_matrix[1,1] = np.sum((ut2[:,2]**2 + ut2[:,0]**2)*
                                    dist_term**(-3./2.)*ut1+1)
    curvature_matrix[2,2] = np.sum((ut2[:,1]**2 + ut2[:,0]**2)*
                                    dist_term**(-3./2.)*ut1+1)
    curvature_matrix[3,3] = len(stations_local)
    curvature_matrix[0,1] = np.sum(-ut1*ut2[:,0]*ut2[:,1]*
                                   dist_term**(-3./2.))
    curvature_matrix[0,2] = np.sum(-ut1*ut2[:,0]*ut2[:,2]*
                                   dist_term**(-3./2.))
    curvature_matrix[1,2] = np.sum(-ut1*ut2[:,1]*ut2[:,2]*
                                   dist_term**(-3./2.))
    curvature_matrix[3,0] = np.sum(-ut2[:,0]*dist_term**(-1./2.))
    curvature_matrix[3,2] = np.sum(-ut2[:,2]*dist_term**(-1./2.))
    curvature_matrix[3,1] = np.sum(-ut2[:,1]*dist_term**(-1./2.))
    curvature_matrix[1,0] = curvature_matrix[0,1]
    curvature_matrix[2,0] = curvature_matrix[0,2]
    curvature_matrix[2,1] = curvature_matrix[1,2]
    curvature_matrix[0,3] = curvature_matrix[3,0]
    curvature_matrix[2,3] = curvature_matrix[3,2]
    curvature_matrix[1,3] = curvature_matrix[3,1]
    # Find the error terms from the matrix solution
    errors = (np.linalg.inv(curvature_matrix/
             (c0*c0*timing_error*timing_error)))**0.5
    return np.array([errors[0,0],errors[1,1],errors[2,2],errors[3,3]] )


 
def mapped_plot(plot_this,from_this,to_this,with_this,dont_forget,
                xmin,xmax,xint,ymin,ymax,yint,location):
    """Make plots in map projection with km color scales, requires input 
       array, lower color scale limit, higher color scale limit, color scale, 
       station location overlay, and array x-,y- min, max and interval, the 
       location of the array center (in lat and lon). Output plot also 
       contains 100, 200 km range rings
    """
    domain =(xmax-xint/2.)
    # maps = Basemap(projection='laea',lat_0=location[0],lon_0=location[1],
    #                width=domain*2,height=domain*2)
    

    s = plt.pcolormesh(np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                       np.arange(ymin-yint/2.,ymax+3*yint/2.,yint)+domain,
                       plot_this, cmap = with_this)
    s.set_clim(vmin=from_this,vmax=to_this)
    #ticks = np.linspace(from_this, to_this, 13, endpoint=True)##这一句就是colorbar的刻度值
    plt.colorbar()

    plt.scatter(dont_forget[:,0]+domain, dont_forget[:,1]+domain, color='k',s=10)
    # maps.drawstates()
    circle=plt.Circle((domain,domain),50000,color='k',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),100000,color='k',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),150000,color='w',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)    
    circle=plt.Circle((domain,domain),200000,color='k',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    
    
    
def mapped_plot2(plot_this,from_this,to_this,with_this,dont_forget,
                xmin,xmax,xint,ymin,ymax,yint,location,uneven_levels):
    """Make plots in map projection with km color scales, requires input 
       array, lower color scale limit, higher color scale limit, color scale, 
       station location overlay, and array x-,y- min, max and interval, the 
       location of the array center (in lat and lon). Output plot also 
       contains 100, 200 km range rings
    """
    
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    domain = (xmax-xint/2.)
    maps = Basemap(projection='laea',lat_0=location[0],lon_0=location[1],
                   width=domain*2,height=domain*2)
    
    
    #uneven_levels = [0, 0.05, 0.1, 0.5,1, 5]
    colors = with_this(np.linspace(0, 1, len(uneven_levels) - 1))
    cmap, norm = mcolors.from_levels_and_colors(uneven_levels, colors)
    
    
    
    
    plt.pcolormesh(np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                       np.arange(ymin-yint/2.,ymax+3*yint/2.,yint)+domain,
                       plot_this, cmap = cmap, norm=norm)
    #s.set_clim(vmin=from_this,vmax=to_this)
    #ticks = np.linspace(from_this, to_this,0.5, endpoint=True)##这一句就是colorbar的刻度值
    #plt.colorbar(ticks=uneven_levels)
    plt.colorbar(ticks=[0,0.05, 0.1 ,0.5,1,1.5,2,3])

    #clb.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    #clb.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.scatter(dont_forget[:,0]+domain, dont_forget[:,1]+domain, color='k',s=10)
    maps.drawstates()
    circle=plt.Circle((domain,domain),50000,color='w',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),100000,color='k',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),150000,color='w',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)    
    circle=plt.Circle((domain,domain),200000,color='k',fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)    

def nice_plot(data,xmin,xmax,xint,centerlat,centerlon,stations,color,cmin,
              cmax,levels_t):
    """Make plots in map projection, requires input array, x-min max and 
       interval (also used for y), center of data in lat, lon, station 
       locations, color scale, min and max limits and levels array for color
       scale.
    """
    domain = xmax-xint/2.
    maps = Basemap(projection='laea',lat_0=centerlat,lon_0=centerlon,
                   width=domain*2,height=domain*2)
    s = plt.pcolormesh(np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                       np.arange(xmin-xint/2.,xmax+3*xint/2.,xint)+domain,
                       data, cmap = color)
    s.set_clim(vmin=cmin,vmax=cmax)
    CS = plt.contour(np.arange(xmin,xmax+xint,xint)+domain,
                     np.arange(xmin,xmax+xint,xint)+domain,
                     data, colors='k',levels=levels_t)
    plt.clabel(CS, inline=1, fmt='%1.2f',fontsize=8)
    plt.scatter(stations[:,0]+domain, stations[:,1]+domain, color='k',s=2)
    maps.drawstates()
    fig = plt.gcf()
    circle=plt.Circle((domain,domain),50000,color='0.5',fill=False)
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),100000,color='0.5',fill=False)
    fig.gca().add_artist(circle)
    circle=plt.Circle((domain,domain),200000,color='0.5',fill=False)
    fig.gca().add_artist(circle)
