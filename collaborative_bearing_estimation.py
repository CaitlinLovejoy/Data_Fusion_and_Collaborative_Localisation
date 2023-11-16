#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collaborative Bearing Estimation

@author: bzamani

two agents use noisy bearing angle measurements and estimate the 2D position of a target
 - they have to fuse their measurements to get an observable situation
 - compare CI with CCE
 
"""

import Utilities
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import datetime
import Gamma_and_gamma as gg 
from scipy.stats import chi2

# # agent 1 and 2 list of positions with initial position added (same starting position for both)
# x1=[np.array([[-10],[10]])]
# x2 =[np.array([[-10],[10]])]

# # target
# x = np.array([[0],[0]])


# v1 = 10 # linear velocity
# th1 = np.deg2rad(45) #angular veloccity

# v2 = 10
# th2 = np.deg2rad(-45)


# dt = .1 
# T = 10

def meas_ellipse (x, th_t, r_s, r_l, th_e):
    
    ''' calculate for a target position an ellipse center and covariance matrix (and its inv) 
    based on  a bearing angle measurement th_t, and sensor characteristics
    min and max range of detection and bearing angle error'''
   
    
    wr = (r_l - r_s)/2 #width radius of ellipse
    d_r = r_s + wr # relative distance of center to x
    hr = d_r * np.tan(th_e) # height radius of ellipse (determinied based on bearing error)
    c = x + np.array([[np.cos(th_t)], [ np.sin(th_t)]]) * d_r  #center
    
    c, P, P_inv = Utilities.ellipse_coordinate2standard (c, wr, hr, th_t)
    
    # print ("measurement ellipse : c, w, h, yaw", c, wr, hr, th_t)
    return c, P, P_inv

class Observer:
    def __init__(self, ID, x0, s, a, r_s, r_l, th_e, c, P, P_inv, method):
        
        # agent parameters
        self.ID = ID
        self.x_l = [x0]
        self.s = s # forward speed
        self.s_a = a #steering angle
        
        # sensor parameters
        self.r_s = r_s # min est range
        self.r_l = r_l # max est range
        self.th_e = th_e # est angle error std 
        
        #estimator parameters
        
        self.c = c #initial guess for where the target is
        self.P = P # initial covariance of target estimate
        self.P_inv = P_inv
        self.fuse_method = method
        
        self.c_l = [c]
        self.P_l = [P]
        self.err_l = []
        self.conf_l=[]
        self.nees_l=[]
        self.anees_l=[]
        self.anees_l_moving=[]
        self.solvable=[]
        self.noiseFloor_l=[]
        self.avnoiseFloor_l=[]
        self.FloorReach=[]
        
        #plot parameters
        if self.ID ==1:
            self.col_p = 'r'
            self.col_e = 'orange'
        else:
            self.col_p = 'b'
            self.col_e = 'k'
        
    def move(self, dt):
        ''' calculates a new position based on the speed and stearing angle and appends it to the trajectory'''
        
        x_n = self.x_l[-1] + dt * np.array([[np.cos(self.s_a)],[np.sin(self.s_a)]]) * self.s
        self.x_l.append(x_n)
        
    def sense(self, x_t, e):
        ''' first calculates a noisy bearing angle measurement towards target
        then produces an uncertainty ellipse for this measurement based on this angle.
        if target is not within sensing range will return None as fourth output or otherwise 1'''
        
        dist = np.linalg.norm(x_t - self.x_l[-1]) 
        
        if dist > self.r_l or dist < self.r_s :
            print ("Target out of range for Agent{}".format(self.ID))
            return self.c, self.P, self.P_inv, None
        
        v = x_t - self.x_l[-1]
        th = np.arctan2 (v[1, 0], v[0,0])
        
        th_t = th + e # add zero mean gaussian with standard deviation th_e
        
        c, P, P_inv = meas_ellipse (self.x_l[-1], th_t, self.r_s, self.r_l, self.th_e)
        
        
        return c, P, P_inv, 1
    
    def estimate(self, cm, Pm, Pm_inv, flag =1):
        
        
        if flag != None: #if there is no measurement do nothing
            
            c, P = Utilities.fuse_KF(self.c, self.P_inv, cm, Pm_inv, "Estimation")
            # d_m =  (self.c- cm).T @ np.linalg.inv(self.P + Pm) @ (self.c- cm) # Mahalanobis distance between the two ellipses 
            # alpha = np.minimum( 1 / (d_m), 1) # discount measurement by 1/d if non-overlapping otherwise 1
            
            # # average the two ellipses
            # P_n = np.linalg.inv( self.P_inv + alpha * Pm_inv)
            
            # self.c = P_n @ ( self.P_inv @ self.c + alpha * Pm_inv @ cm)
            # self.P_inv = self.P_inv + alpha * Pm_inv
            # self.P = P_n
            
            self.c = c
            self.P = P
            self.P_inv = np.linalg.inv(P)
            
            self.c_l.append(self.c)
            self.P_l.append(self.P)
        
    def propagate(self, t):
        
        ''' deflates the confidence in estimation in a symmetric and based on time passed'''
        self.P = self.P * (1 + t/100) 
        self.P_inv = np.linalg.inv(self.P)
        
    
    def est_err (self, x_t):
        
        # e = (self.c- x_t).T @ self.P_inv @ (self.c - x_t) # Mahalanobis distance of true target from our estimate
        # e = e[0,0]
        
        e = np.linalg.norm(self.c- x_t) # Euclidean distance
      
        self.err_l.append(e)
        
        conf = np.linalg.det(self.P)
        self.conf_l.append(conf)

    def nees (self, x_t):
        #normalised estimation error squared 
        e = (self.c - x_t)
        
        # e = np.linalg.norm(self.c- x_t)
        
        # e_mag = np.linalg.norm(self.c - x_t)
        # e[0] = e[0]/e_mag
        # e[1]= e[1]/e_mag
        # eT = np.linalg.norm(self.c- x_t)
        eT = e.transpose()
        

        # arrmax, arrmin = self.P_inv.max(), self.P_inv.min()
        # P_inv_calc = (self.P_inv - arrmin) / (arrmax - arrmin)
        # P_inv_det = np.linalg.det(self.P_inv)
        # P_inv_calc = self.P_inv/P_inv_det
        # norm = np.linalg.norm(self.P_inv)
        # P_inv_calc = self.P_inv / norm


        # print("P1_inv normalised", P_inv_calc)
        # p_inv = np.linalg.inv(self.P)
        # nees_m = eT @ P_inv_calc @ e
        # nees_m = eT @ self.P_inv @ e
        nees_m = eT @ self.P_inv @ e
        # nees_m = eT @ self.P @ e
        nees = nees_m[0]
        # mean, var, skew, kurt = chi2.stats(nees, moments='mvsk')
        # print(mean)
        self.nees_l.append(nees)
        # self.nees_l.append(nees_m)
        # print("NEES", nees)

    def anees(self, simulation_time):
        samples = simulation_time+1
        multiplier = 1/(samples)
        summed_nees = sum(self.nees_l)
        anees = summed_nees*multiplier
        self.anees_l.append(anees)
    
    def anees_moving_frame(self, simulation_time):
        samples = 10
        multiplier = 1/(samples)
        sum_start = simulation_time-samples 
        averaging_list =[]
        for i in range(samples):
            val = self.nees_l[sum_start+i]
            averaging_list = np.append(averaging_list, val)
        
        summed_nees = sum(averaging_list)
        anees = summed_nees*multiplier
        self.anees_l_moving.append(anees)

    def noiseFloor(self, simulation_time):
        valA = self.err_l[simulation_time-1]
        valB = self.err_l[simulation_time]
        val = valA-valB
        self.noiseFloor_l.append(val)

    def avgnoiseFloor(self, simulation_time):
        samples = simulation_time+1
        multiplier = 1/(samples)
        # sum_start = simulation_time+1
        # averaging_list =[]
        # for i in range(samples):
        #     val = self.noiseFloor_l[sum_start+i]
        #     averaging_list = np.append(averaging_list, val)
        summed = sum(self.noiseFloor_l)
        av = summed*multiplier
        if abs(av) < 0.015:
            # if len(self.FloorReach) == 0:
            self.FloorReach.append(simulation_time)
        self.avnoiseFloor_l.append(av)
        

    def fuse(self, agent):
        
        if self.fuse_method == 'CCE':
            self.c, self.P = Utilities.fuse_CCE_opt(self.c, self.P_inv, agent.c, agent.P_inv)
            self.P_inv = np.linalg.inv(self.P)
            self.c_l.append(self.c)
            self.P_l.append(self.P)
            
        if self.fuse_method =='CI':
            self.c, self.P = Utilities.fuse_CI_opt(self.c, self.P_inv, agent.c, agent.P_inv)
            self.P_inv = np.linalg.inv(self.P) 
            self.c_l.append(self.c)
            self.P_l.append(self.P)
        
        if self.fuse_method == "Kalman":
            
            self.c, self.P = Utilities.fuse_KF(self.c, self.P_inv, agent.c, agent.P_inv, "Fusion")
            self.P_inv = np.linalg.inv(self.P) 
            self.c_l.append(self.c)
            self.P_l.append(self.P)
        
        if self.fuse_method == "ICI":
            
            self.c, self.P = Utilities.fuse_ICI_opt(self.c, self.P_inv, agent.c, agent.P_inv)
            self.P_inv = np.linalg.inv(self.P) 
            self.c_l.append(self.c)
            self.P_l.append(self.P)
        
        if self.fuse_method == "EI":
            
            solveFlag = gg.checkSol(self.P_inv)
            self.solvable.append(solveFlag)
            self.c, self.P = Utilities.fuse_EI_opt(self.c, self.P_inv, agent.c, agent.P_inv, 0.0001, solveFlag)
            self.P_inv = np.linalg.inv(self.P) 
            self.c_l.append(self.c)
            self.P_l.append(self.P)
            
                
            
            
        
    def plot_pos(self, ax, label=False):
        if self.ID ==1:
            color = 'tab:red'
            marker = '>'
        else:
            color = 'tab:blue'
            marker = 'v'
        
        if label:    
            ax.scatter(self.x_l[-1][0,0], self.x_l[-1][1,0], s = 100, marker=marker, alpha=.9, c = color, label = "Agent{}".format(self.ID))
        else:
            ax.scatter(self.x_l[-1][0,0], self.x_l[-1][1,0], s = 100, marker=marker , alpha=.5, c = color )
    def plot_meas(self, cm, Pm, ax, label=False):   
        
        if self.ID ==1:
            color = 'tab:red'
        else:
            color = 'tab:blue'
            
        c, w_r, h_r, yaw  = Utilities.ellipse_standard2coordinates (cm, Pm)
        if label == True:
            Utilities.plot_ellipse (ax, c, w_r, h_r, yaw, color,  'a{}-meas'.format(self.ID), alpha=.1, ls='-', label =label)
            
        else:
            Utilities.plot_ellipse (ax, c, w_r, h_r, yaw, color, 'a{}-meas'.format(self.ID), alpha=.01, ls= 'dashed')#(0, (5, 10)))
            # Utilities.plot_ellipse (ax, c, w_r, h_r, yaw, color, alpha=.01, ls= 'dashed')
    
    def plot_est(self, ax, label):
        
    
        if self.fuse_method == "CCE":
            color = 'b'
        if self.fuse_method == "CI":
            color = "r"
        if self.fuse_method == "ICI":
            color = "c"
        if self.fuse_method == "EI":
            color = "mediumorchid"
            if self.ID ==1:
                color = "mediumorchid"
            else:
                color = "darkorange"
        if self.fuse_method == "Kalman":
            color ='g'
        if self.fuse_method == None:
            color = 'k'
            
        if self.ID ==1:
            ls = "-"
        
        else:
            ls = "--"
            # color = "k"
            
            
            
            
            
        c, w_r, h_r, yaw  = Utilities.ellipse_standard2coordinates (self.c, self.P)
        Utilities.plot_ellipse (ax, c, w_r, h_r, yaw, color, 'a{0}{1}'.format(self.ID, self.fuse_method), alpha=.8, ls= ls, label=True)
        # Utilities.plot_ellipse (ax, c, w_r, h_r, yaw, color, name=label, alpha=0.75, ls= ls, label=True)
    
            
            
            
       
        
        
        



    
    
