#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A number of methods for attitude and pose representations and conversions

@author: BehzadZ
"""

import numpy as np


def quat_inv(q):
    """calculates the inverse of a unit quaternion as its conjugate.
    q: 4x1 np ndarray unit norm quaternion
    
    -> 4x1 np ndarray unit norm quaternion
    """
    
    return np.concatenate([q[0:1], -q[1:]])


def quat2angle(q1, q2=np.array([[1],[0],[0],[0]])):
    """Calculates the angle between 2 unit quaternions if one is the 'zero' quaternion,
     np.array([[1],[0],[0],[0]], it gives the angle of the first quaternion.
    
    q1: 4x1 np ndarray unit norm quaternion
    q2: 4x1 np ndarray unit norm quaternion
    
    -> 1x1 np ndarray, angle between the two quaternions
    """

    angle = 2 * np.arccos(np.absolute(np.dot(q1.T, q2)));

    if np.imag(angle)!=0:
        print ('Error: quaternion is not unit norm')

    return angle

 
def ang_vel2quat (w, dt):
    """Calculates the quaternion rotation induced by dt time angular
    velocity w frw integ
    w: 3x1 np ndarray of angular velocity
    dt: scalar time step of forward integration
    
    -> 4x1 np ndarray unit norm quaternion
    """
    
    # the assumed constant angular velocit
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm(w)
#   
    if (w_s < 10e-9):
        return np.array([[1],[0],[0],[0]])
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        
        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        q_w = np.concatenate([np.array([[np.cos(w_s/2)]]), np.sin(w_s/2) * w_ax])
    
        return q_w

    
def ang_vel2Rotation (w, dt):
    """Calculates the rotation matrix induced by dt time angular velocity w
    frw integ
    w: 3x1 np ndarray of angular velocity
    dt: scalar time step of forward integration
    
    -> 3x3 ndarray Rotation matrix np ndarray 
    """
    
    # the assumed constant angular velocit
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm(w)
#   
    if (w_s < 10e-9):
        return np.eye(3)
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        
        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        R_w = np.eye(3) + np.sin(w_s) * skew_matrix(w_ax) + (1-np.cos(w_s)) * np.dot (skew_matrix(w_ax), skew_matrix(w_ax))
    
        return R_w
    
    
    
def Rotation2angle (R):
    """Gives angle of rotation of a rotation matrix.
    R : 3x3 np array orthonormal with det=1 (rotation)
    
    -> angle of rotation 
    """
    
    theta = np.arccos(.5 * np.trace(R) - .5)
    
    return theta


def Rotation2axis(R):
    """Gives (unit norm) axis of rotation of a rotation matrix.
    R : 3x3 np array orthonormal with det=1 (rotation)
    
    -> axis of rotation 
    """
    
    theta = Rotation2angle(R)
    
    if (theta < 10e-9):
        # random direction
        axis= np.array([[1],[0],[0]])
        return axis
    
    else:  
        axis = (1 / (np.sin(theta))) * skew2vec(projSkew(R))
    
    return axis


def Rotation2quaternion(R):
    """Calculates unit quaternion equivalent of roration matrix R
    R : 3x3 np ndarray orthonormal with det=1 (rotation)
    
    -> 4x1 unit quaternion np ndarray 
    """
    
    theta = Rotation2angle(R)

    axis = Rotation2axis(R)
    
    quaternion = np.concatenate([np.reshape(np.array([np.cos(theta/2)]), (1,1)) , np.sin(theta/2)*axis])

    return quaternion


def quat2Rotation(q):
    """Calculates a rotation matrix equivalent of unit quaternion q (nonunique with sign of first element)
    q : 4x1 unit quaternion np ndarray 
    
    -> 3x3 np ndarray orthonormal with det=1 (rotation)
    """
    
    w = q[1:]
    w_x = skew_matrix(w)

    R = np.eye(3) + 2 * q[0] * w_x + 2 * np.dot (w_x, w_x)

    return R


def quat_mult (q1, q2):
    """Multiply q2 from left by q1
    
    q1: 4x1 np ndarray, first unit norm quaternion
    q2: 4x1 np ndarray, second unit norm quaternion
    
    -> 4x1 np ndarray, unit norm quaternion
    """
    
    q_0 = q1[0:1] * q2[0:1] - np.dot(q1[1:].T, q2[1:])

    q_v = q1[0:1] * q2[1:] + q2[0:1] * q1[1:] + np.cross(q1[1:], q2[1:], axis=0)
    
    return np.concatenate ([q_0, q_v])


def skew_matrix (w):
    """ Calculates the skew symmetric version of angular velocity w
    
    w: 3x1 np ndarray, angular velocity vector
    
    -> 3x3 np ndarray
    """
    w = np.reshape(w, (3,1))
   
    S = np.array([ [0, -w[2,0], w[1,0]], [w[2,0], 0, -w[0,0]], [-w[1,0], w[0,0], 0] ])
    return S

def vec2skew (w):
    """ Calculates the skew symmetric version of angular velocity w
    
    w: 3x1 np ndarray, angular velocity vector
    
    -> 3x3 np ndarray
    """
    w = np.reshape(w, (3,1))  
  
    S = np.array([ [0, -w[2,0], w[1,0]], [w[2,0], 0, -w[0,0]], [-w[1,0], w[0,0], 0] ])
    return S

def skew2vec (S):
    """ Extracts the vector out of a skew symmetric matrix S. 
    
    S: 3x3 np ndarray
    
    -> 3x1 np ndarray, angular velocity vector
    """
     
    w = np.reshape(np.array([ S[2,1], S[0,2], S[1,0] ]),(3,1))
    
    return w

def ang_vel2Rp(w, dt):
    """Calculates the rotation multiplier of position p for the exponential map of SE(3)
    w: 3x1 np ndarray of angular velocity
    dt: 1x1 time step of forward integration
    
    -> 3x3 Rotation matrix np ndarray 
    """
    # the assumed constant angular velocit
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm(w)
#   
    if (w_s < 1e-14):
        return np.eye(3)
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        
        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        R_w = np.eye(3) + ((1-np.cos(w_s))/w_s) * skew_matrix(w_ax) + ((w_s- np.sin(w_s))/w_s) * np.dot(skew_matrix(w_ax), skew_matrix(w_ax)) 
    
        return R_w
    
def ang_vel2RpInv (w, dt):
    """Calculates the rotation multiplier of position p for the log map of SE(3)
    w: 3x1 np ndarray of angular velocity
    dt: 1x1 time step of forward integration
    
    -> 3x3 Rotation matrix np ndarray 
    """
    # the assumed constant angular velocit
    w = w * dt
    
    # The angle of rotation induced by this velocity
    w_s = np.linalg.norm (w)
    
    if (w_s < 1e-14):
        return np.eye(3)
    else:
        # The axix of rotation induced by this velocity
        w_ax = (1 / w_s) * w
        

        # the quaternion version of the angle and axis of rotation (note that as always quaternion takes half the angle)
        R_w_inv = np.eye(3) - w_s * skew_matrix(w_ax)/2 + (1- w_s * np.sin(w_s)/(2 - 2*np.cos(w_s))) * np.dot(skew_matrix(w_ax), skew_matrix(w_ax)) 
    
        return R_w_inv
    

def ref2body(q, p, w):
    """ transform vector w expressed in ref coordinates to body fixed coordinates of robot with
    pose (q,p).
    q: 4x1 np ndarray, unit norm quaternion    
    p: 3x1 np ndarray, position vector    
    w: 3x1 np ndarray, vector
    
    -> 3x1 np ndarray
    """            
    
    return np.dot(quat2Rotation(q).T , w - p)


def body2ref(q, p, w):
    """ transform vector w expressed in  body fixed coordinates of robot with
    pose (q,p) to ref coordinates.
    q: 4x1 np ndarray, unit norm quaternion    
    p: 3x1 np ndarray, position vector    
    w: 3x1 np ndarray, vector
    
    -> 3x1 np ndarray
    """
    return np.dot(quat2Rotation(q) , w) + p

def  projsym(M):
    """Calculates the symmetric projection of matrix M.
    M: nxn np ndarray
    
    -> nxn np ndarray
    """

    return .5 * (M + M.T)

def  projSkew(M):
    """Calculates the skew symmetric projection of matrix M.
    M: nxn np ndarray
    
    -> nxn np ndarray
    """

    return .5 * (M - M.T)

def eigsorted(cov):
    """Calculates descending ordered eigen values and associated eigen vectors of symmetrix matrix cov.
    cov: 2x1 np ndarray
    
    -> 2x1 np array, 2x2 np ndarray
    """
    
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def cov2ellipse(cov):
    """Calculates radius of two axis and orientation angle of ellipse representing a convarinace matrix.
    cov: 2x1 np ndarray
    
    -> scalar radius of major axis, scalar radius of minor axis, yaw of the ellipse
    """
    
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * np.sqrt(vals)
    return w,h,theta

    
def integrate_forward_quaternion (w, quaternion, dt):
    """Implements the exponential map of quaternions explicitly.
    w: 3x1 np ndarray of angular velocity
    quaternion: 4x1 unit quaternion np ndarray (current quaternion) 
    dt: 1x1 time step of forward integration
    
    -> 4x1 unit quaternion np ndarray (next quaternion) 
    """ 

    
    q_w = ang_vel2quat(w, dt)
    
    # the new quaternion is obtained by multiplying the last updated quaternion with the one obtain from the velocity 
    q = quat_mult (quaternion,  q_w)
 
    return q




def integrate_forward_position(w, v, p, q, dt):
    """Implements the exponential map of SE(3) explicitly.
    w: 3x1 np ndarray of angular velocity
    v: 3x1 np ndarray of linear velocity
    p: 3x1 np ndarray previous robot positon
    quaternion: 4x1 unit quaternion np ndarray (current robot quaternion) 
    dt: 1x1 time step of forward integration
    
    -> 3x1 np ndarray of next robot position 
    """
    R_w = ang_vel2Rp(w, dt)

    return body2ref(q,p, dt * np.dot(R_w , v))  

def pose_error(q1, p1, q2, p2):
    """calculates the difference between two poses.
    q1: 4x1 np ndarray, unit norm quaternion    
    p1: 3x1 np ndarray, position vector  
    q2: 4x1 np ndarray, unit norm quaternion    
    p2: 3x1 np ndarray, position vector
    
    -> scalar (error(pose1-pose2))
    """
    
    return 2 * quat2angle(q2, q1) + np.linalg.norm(ref2body(q2, p2, p1)) 

def pose2velocity(q,p):
    """Implements the logarithm map of SE(3) explicitly.
    q: 4x1 np ndarray, unit norm quaternion    
    p: 3x1 np ndarray, position vector    
    
    -> 3x1, 3x1 np ndarrays of angular velocity and linear velocity
    """
    
    R = quat2Rotation(q)
    theta = Rotation2angle(R)
    axis = Rotation2axis(R)
    
    ang_vel = theta * axis
    
    V_inv = ang_vel2RpInv (ang_vel, 1)
    
    lin_vel = np.dot (V_inv , p) 
    
    return ang_vel, lin_vel
    
    
