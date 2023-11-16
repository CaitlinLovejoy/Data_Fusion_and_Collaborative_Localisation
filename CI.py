# -*- coding: utf-8 -*-
"""

- Methods for plotting uncertainty ellipses of a measurement or prior estimate
- Methods for coverting between different ellipse parameterisations 
- methods for fusion of various algorithms including Kalman, CI, ICI, BSI and SDP  
- methods for fusing prior estimate and measurement uncertainty ellipses, 2 prior estimate/elipses 

- all in 2D 
@author: BZ
"""


import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import datetime
from scipy.optimize import fminbound
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import Geometry




def plot_ellipse (ax, c1, w1_r, h1_r, yaw1, color='r', name = "ellipse 1", alpha=1, label=False, ls="-"):
    
    
    ellipse1 = Ellipse(xy=(c1[0,0], c1[1,0]), width= 2*w1_r, height = 2*h1_r, angle= np.rad2deg(yaw1), 
                        edgecolor=color, fc='None', lw=2, ls=ls, alpha=alpha)
    ax.add_artist(ellipse1)
    ellipse1.set_clip_box(ax.bbox)
    
    if label:
        ax.scatter(c1[0,0],c1[1,0], s = 50, marker='^', c = color, label = name, alpha=alpha)
    else:
        ax.scatter(c1[0,0],c1[1,0], s = 50, marker='^', c = color, alpha=alpha)

    
def ellipse_coordinate2standard (center, width_radius, height_radius, yaw):

    


    #rotation matrix based on theta
    R = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
    # eigen square root value matrix of ellipse
    D = np.array([[width_radius, 0],[0, height_radius]])
    # inverse of D
    D_inv = np.array([[1/width_radius, 0],[0, 1/height_radius]])

    # shape matrix of ellipse
    P = R @ D @ D @ R.T

    P_inv = R @ D_inv @ D_inv @ R.T
    
    return center, P, P_inv

def ellipse_standard2coordinates (center, P):
    
    
    u, s, vh = np.linalg.svd(P, full_matrices=False, compute_uv=True, hermitian=True)
    
    
    width_radius = s[0]**.5
    height_radius = s[1]**.5
    yaw = np.arctan2(u[1,0],u[0,0])
    
    
    return center, width_radius, height_radius, yaw


    
    
def fuse_SDP (c1, P1_inv, c2, P2_inv): 
    n = np.shape(c1)[0]
    F = cp.Variable((n, n), PSD= True)
    # constr_0 = [F >> 0]
    b = cp.Variable((n, 1))
    t = cp.Variable((2, 1))
    
    F1 = P1_inv
    g1 = - F1 @ c1
    h1 =  - c1.T @ g1 - 1
    
    F2 = P2_inv
    g2 = - F2 @ c2
    h2 =  - c2.T @ g2 - 1
    
    
    
    Fh1 = cp.hstack([F - t[0,0]*F1 -t[1,0]*F2, b - t[0,0]*g1 -t[1,0]*g2])
    
    Fh2 = cp.hstack([b.T - t[0,0]*g1.T -t[1,0]*g2.T, -1 - t.T @ np.array([[h1[0,0]], [h2[0,0]]])])
    
    Fh = cp.vstack([Fh1, Fh2])
    
    
    bh = cp.vstack ([np.zeros((n, n)), b.T])
    
    
    LMI = cp.vstack([cp.hstack([Fh, bh]), cp.hstack([bh.T , -F])]) 
    
   
    
    constr_1 = [LMI << 0]
    constr_2 =  [t[0,0] >= 0]
    constr_3 = [t[1,0]>= 0]
        
    obj = cp.Minimize(- cp.log_det(F))
    
    
    
    prob = cp.Problem(obj,   constr_1 + constr_2 + constr_3 )
    
    # conic sdp needs SCS or cvxOPT if formulated differently
    prob.solve(solver = cp.SCS)
    
    print ("status of spd: ", prob.status)
    
    
    
    P = np.linalg.inv(F.value)
    c = - P @ b.value
    
    print ("det of SDP = ", np.linalg.det(P))
    c = - P @ b.value
    
    return c, P
   

    
def fuse_CI (c1, P1_inv, c2, P2_inv, alpha): 
    
    P_inv = alpha * P1_inv + (1-alpha) * P2_inv
    
    P = np.linalg.inv(P_inv)
    
    c = P @ (alpha * P1_inv @ c1 + (1-alpha) * P2_inv @ c2)
    print ("det of CI = ", np.linalg.det(P))
    return c, P



def fuse_BSI (c1, P1_inv, c2, P2_inv, alpha): 
    
    P_inv = alpha * P1_inv + (1-alpha) * P2_inv
    P = np.linalg.inv(P_inv)
    k = 1 - alpha * (1-alpha) * (c2 -c1).T @  P2_inv @ P @  P1_inv @ (c2 - c1)
    P_t = k * P 

    
    c = P @ (alpha * P1_inv @ c1 + (1-alpha) * P2_inv @ c2)
    print ("det of BSI = ", np.linalg.det(P_t))
    return c, P_t

def fuse_ICI_opt (c1, P1_inv, c2, P2_inv): 
    P1 = np.linalg.inv(P1_inv)
    P2 = np.linalg.inv(P2_inv)

    def optimize_fn(alpha):
        P_det = np.linalg.det(np.linalg.inv(P1_inv + P2_inv - np.linalg.inv(alpha * P1 + (1-alpha) * P2)))
        return P_det
    alpha = fminbound(optimize_fn, 0, 1)
    
    P_inv = P1_inv + P2_inv - np.linalg.inv(alpha * P1 + (1-alpha) * P2)
    
    P = np.linalg.inv(P_inv)
    print ("det of ICI opt = ", np.linalg.det(P))
    
    c = P @ ((P1_inv - alpha * np.linalg.inv(alpha * P1  + (1-alpha) * P2)) @ c1 +
             (P2_inv - (1-alpha) * np.linalg.inv(alpha * P1  + (1-alpha) * P2)) @ c2)
    
    return c, P

def fuse_CI_opt (c1, P1_inv, c2, P2_inv): 
    
    if check_overlap(c1, P1_inv, c2, P2_inv):
        
        def optimize_fn(alpha):
            return np.linalg.det(np.linalg.inv(alpha * P1_inv + (1-alpha) * P2_inv))
        alpha = fminbound(optimize_fn, 0, 1)
        
        P_inv = alpha * P1_inv + (1-alpha) * P2_inv
        
        P = np.linalg.inv(P_inv)
        print ("det of CI opt = ", np.linalg.det(P))
        
        c = P @ (alpha * P1_inv @ c1 + (1-alpha) * P2_inv @ c2)
        
        return c, P
    else:
        # return c1, np.linalg.inv(P1_inv)
        print ("CI: averaged with discounted second")
        
        P1 = np.linalg.inv(P1_inv)
        P2 = np.linalg.inv(P2_inv)
        
        d_m =  (c1- c2).T @ np.linalg.inv(P1 + P2) @ (c1- c2) # Mahalanobis distance between the two ellipses 
        alpha = np.minimum( 1 / (d_m), 1) # discount measurement by 1/d if non-overlapping otherwise 1
        
        # average the two ellipses
        P = np.linalg.inv( P1_inv + alpha * P2_inv)
        
        c = P @ ( P1_inv @ c1 + alpha * P2_inv @ c2)
        
        return c, P

def fuse_BSI_opt (c1, P1_inv, c2, P2_inv): 
    
    if check_overlap(c1, P1_inv, c2, P2_inv):
    
        def optimize_fn(alpha):
            P = np.linalg.inv(alpha * P1_inv + (1-alpha) * P2_inv)
            k = 1 - alpha * (1-alpha) * (c2 -c1).T @  P2_inv @ P @  P1_inv @ (c2 - c1)
            P_t = k * P 
            return np.linalg.det(P_t)
        alpha = fminbound(optimize_fn, 0, 1)
        
        
        P_inv = alpha * P1_inv + (1-alpha) * P2_inv
        P =  np.linalg.inv(P_inv)
        
        c = P @ (alpha * P1_inv @ c1 + (1-alpha) * P2_inv @ c2)
        
        k = 1 - alpha * (1-alpha) * (c2 -c1).T @  P2_inv @ P @  P1_inv @ (c2 - c1)
        P_t = k * P
        print ("det of BSI opt = ", np.linalg.det(P_t))
        return c, P_t
    else:
        # return c1, np.linalg.inv(P1_inv)
        print ("BSI: averaged with discounted second")
        P1 = np.linalg.inv(P1_inv)
        P2 = np.linalg.inv(P2_inv)
        
        d_m =  (c1- c2).T @ np.linalg.inv(P1 + P2) @ (c1- c2) # Mahalanobis distance between the two ellipses 
        alpha = np.minimum( 1 / (d_m), 1) # discount measurement by 1/d if non-overlapping otherwise 1
        
        # average the two ellipses
        P = np.linalg.inv( P1_inv + alpha * P2_inv)
        
        c = P @ ( P1_inv @ c1 + alpha * P2_inv @ c2)
        
        return c, P

def fuse_KF (c1, P1_inv, c2, P2_inv, process="Fusion" ): 
    
    if process == "Estimation":
        P_inv =  P1_inv +  P2_inv
        
        P = np.linalg.inv(P_inv)
       
        
        c = P @ ( P1_inv @ c1 +  P2_inv @ c2)
        # P = P * 1.1 
        
        return c, P
    
    if check_overlap(c1, P1_inv, c2, P2_inv):
        P_inv =  P1_inv +  P2_inv
        
        P = np.linalg.inv(P_inv)
       
        
        c = P @ ( P1_inv @ c1 +  P2_inv @ c2)
        # P = P * 1.1 
        
        return c, P
    else:
        print ("Kalman-{}: averaged with discounted second".format(process))
        P1 = np.linalg.inv(P1_inv)
        P2 = np.linalg.inv(P2_inv)
        
        d_m =  (c1- c2).T @ np.linalg.inv(P1 + P2) @ (c1- c2) # Mahalanobis distance between the two ellipses 
        alpha = np.minimum( 1 / d_m, 1) # discount measurement by 1/d if non-overlapping otherwise 1
        
        # average the two ellipses
        P = np.linalg.inv( P1_inv + alpha * P2_inv)
        
        c = P @ ( P1_inv @ c1 + alpha * P2_inv @ c2)
        
        return c, P
        

def plot_ellipsoid_3d(c, A, ax, color, t= 'solid'):
    """Plot 3D Ellipsoid on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))
    
    ax.scatter (c[0,0],c[1,0],c[2,0], marker='^', color=color, alpha=1)
    
    for i in range(len(x)):
        for j in range(len(x)):
            a = c + A @ np.array([[x[i,j]],[y[i,j]],[z[i,j]]])
            x[i,j], y[i,j], z[i,j] = tuple(a.reshape(1, -1)[0])
    if t == 'wire':        
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)
    else :
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)
    
def check_overlap (c1, P1_inv, c2, P2_inv):
    ''' checks whether the two ellipses overlap via the Mahalanobis distance between the two ellipses'''
    P1 = np.linalg.inv(P1_inv)
    P2 = np.linalg.inv(P2_inv)
    
    if (c1 - c2).T @ np.linalg.inv(P1 + P2) @ (c1 - c2) >= 2:
        return False
    else:
        return True

test =0
if test :    
    # first ellipse 
    c1 = np.array([[3],[2]])
    w1_r = 4.4
    h1_r = 3.2
    yaw1 = -np.deg2rad(20)
    c1, P1, P1_inv = ellipse_coordinate2standard (c1, w1_r, h1_r, yaw1)
    
    print ("ellipse 1: c, w, h, yaw", c1, w1_r, h1_r, yaw1)
    
    
    # second ellipse
    c2 = np.array([[6.5],[4]])
    w2_r = 4.4
    h2_r = 3.2
    yaw2 = np.deg2rad(90)
    c2, P2, P2_inv = ellipse_coordinate2standard (c2, w2_r, h2_r, yaw2)
    print ("second ellipse ", c2, w2_r, h2_r, yaw2)
    
    
    
    
    begin_time = datetime.datetime.now()
    c3, P3 = fuse_SDP (c1, P1_inv, c2, P2_inv)
    print("microseconds taken for spd program: ", (datetime.datetime.now() - begin_time).microseconds)
    
    c4, P4 = fuse_CI (c1, P1_inv, c2, P2_inv, .5)
    c5, P5 = fuse_KF (c1, P1_inv, c2, P2_inv)
    c6, P6 = fuse_CI_opt(c1, P1_inv, c2, P2_inv)
    c7, P7 = fuse_BSI(c1, P1_inv, c2, P2_inv, 0.5)
    
    begin_time = datetime.datetime.now()
    c8, P8 = fuse_BSI_opt(c1, P1_inv, c2, P2_inv)
    print("microseconds takec for the BSI optimisation= ", (datetime.datetime.now() - begin_time).microseconds)
    
    c9, P9 = fuse_ICI_opt(c1, P1_inv, c2, P2_inv)
    
    
    c3, w3_r, h3_r, yaw3  = ellipse_standard2coordinates (c3, P3)
    c4, w4_r, h4_r, yaw4  = ellipse_standard2coordinates (c4, P4)
    c5, w5_r, h5_r, yaw5 = ellipse_standard2coordinates (c5, P5)
    c6, w6_r, h6_r, yaw6 = ellipse_standard2coordinates (c6, P6)
    c7, w7_r, h7_r, yaw7 = ellipse_standard2coordinates (c7, P7)
    c8, w8_r, h8_r, yaw8 = ellipse_standard2coordinates (c8, P8)
    c9, w9_r, h9_r, yaw9 = ellipse_standard2coordinates (c9, P9)
    # print ("CI ellipse ", c3, w3_r, h3_r, yaw3)
    
    # c3 = np.array([[-2],[1]])
    # w3_r = 3
    # h3_r = .5
    # yaw3 = 2.5
    # c3, P3, P3_inv = ellipse_coordinate2standard (c3, w3_r, h3_r, yaw3)
    plot = 1
    if plot : 
            
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        # x_ls = np.linspace(-10, 10, 100)
        ax.set_xlim(-5, 13)
        ax.set_ylim(-2, 10)
        
        plot_ellipse (ax, c1, w1_r, h1_r, yaw1, 'r', 'e_1, det={}'.format(np.round(np.linalg.det(P1), decimals= 2)), alpha=.9)
        plot_ellipse (ax, c2, w2_r, h2_r, yaw2, 'b', 'e_2, det={}'.format(np.round(np.linalg.det(P2), decimals=2)), alpha=.9)
        
        
        
        
        plot_ellipse (ax, c5, w5_r, h5_r, yaw5, 'g', name ='Kalman, det={}'.format(np.round(np.linalg.det(P5), decimals=2)))
        plot_ellipse (ax, c3, w3_r, h3_r, yaw3 , 'orange', name='SDP, det={}'.format(np.round(np.linalg.det(P3), decimals=2)), alpha =.9)
        plot_ellipse (ax, c8, w8_r, h8_r, yaw8 , 'k',  'BSI_opt, det={}'.format(np.round (np.linalg.det(P8), decimals = 2)))
        plot_ellipse (ax, c6, w6_r, h6_r, yaw6 , 'brown', 'CI_opt, det={}'.format(np.round(np.linalg.det(P6), decimals = 2)))
        plot_ellipse (ax, c4, w4_r, h4_r, yaw4 , 'c', 'CI, det={}'.format(np.round(np.linalg.det(P4), decimals=2)), alpha= .5)
        
        plot_ellipse (ax, c7, w7_r, h7_r, yaw7 , 'm', 'BSI, det={}'.format(np.round(np.linalg.det(P7), decimals=2)), alpha = .5)
        plot_ellipse (ax, c9, w9_r, h9_r, yaw9 , 'orange', 'ICI_opt, det={}'.format(np.round(np.linalg.det(P9), decimals=2)), alpha = .5)
        ax.legend()
           
        # ax.scatter(x[0,0],x[1,0], s = 60, marker='o')
        # ax.scatter(x_left[0,0],x_left[1,0], s = 60, marker='o', c='b')
        # ax.scatter(x_right[0,0],x_right[1,0], s = 60, marker='o', c ='r' )
        # plt.get_current_fig_manager().window.showMaximized()
        
        plt.savefig("all_fusion2.png")
        
        plt.show()
        
        
        # 3D experiment; Do BSI-opt and SDP coincide in this case as well? 
        
        #  first ellipsoid
        #rotation matrix based on theta
        angle_1 = np.deg2rad(20)
        
        axis_1 = np.array([[0], [0], [1]])
        q_1 = np.vstack((np.array([[np.cos(angle_1)]]), np.sin(angle_1)*axis_1))
        
        R_1 = Geometry.quat2Rotation(q_1)
        Radius_1 = [2,3,4]
        # eigen square root value matrix of ellipse
        D_1 = np.array([[Radius_1[0], 0, 0],[0, Radius_1[1], 0],[0, 0, Radius_1[2]]])
        # inverse of D
        D_inv_1 = np.array([[1/Radius_1[0], 0, 0],[0, 1/Radius_1[1], 0],[0, 0, 1/Radius_1[2]]])
        
        # shape matrix of ellipse
        P_1 = R_1 @ D_1 @ D_1 @ R_1.T
        
        P1_inv = R_1 @ D_inv_1 @ D_inv_1 @ R_1.T
        
        c1 = np.array([[0], [0], [0]])
        
        
        #  2nd ellipsoid
        #rotation matrix based on theta
        angle_2 = np.deg2rad(-40)
        axis_2 = np.array([[0], [0], [1]])
        q_2 = np.vstack((np.array([[np.cos(angle_2)]]), np.sin(angle_2)*axis_2))
        
        R_2 = Geometry.quat2Rotation(q_2)
        Radius_2 = [2,3,4]
        # eigen square root value matrix of ellipse
        D_2 = np.array([[Radius_2[0], 0, 0],[0, Radius_2[1], 0],[0, 0, Radius_2[2]]])
        # inverse of D
        D_inv_2 = np.array([[1/Radius_2[0], 0, 0],[0, 1/Radius_2[1], 0],[0, 0, 1/Radius_2[2]]])
        
        # shape matrix of ellipse
        P_2 = R_2 @ D_2 @ D_2 @ R_2.T
        
        P2_inv = R_2 @ D_inv_2 @ D_inv_2 @ R_2.T
        
        c2 = np.array([[1], [1], [1]])
        
        
        fig = plt.figure(figsize=(10., 10.))
        ax = fig.add_subplot(111, projection='3d')
        
        plot_ellipsoid_3d(c1, R_1@D_1, ax, 'r')
        plot_ellipsoid_3d(c2, R_2@D_2, ax, 'b')
        
        
        ax.set_xlim(-10., 10.)
        ax.set_ylim(-10., 10.)
        ax.set_zlim(-10., 10.)
        ax.set_aspect('auto')
        fig.tight_layout()
        
        
        
        begin_time = datetime.datetime.now()
        c3, P3 = fuse_SDP (c1, P1_inv, c2, P2_inv)
        print("microseconds taken for spd program: ", (datetime.datetime.now() - begin_time).microseconds)
        plot_ellipsoid_3d(c3, sp.linalg.sqrtm(P3), ax, 'k')
        
        begin_time = datetime.datetime.now()
        c8, P8 = fuse_BSI_opt(c1, P1_inv, c2, P2_inv)
        print("microseconds takec for the BSI optimisation= ", (datetime.datetime.now() - begin_time).microseconds)
        plot_ellipsoid_3d(c8, sp.linalg.sqrtm(P8), ax, 'y' )
        
        ax.legend()
        plt.show()