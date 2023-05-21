# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:53:10 2023

@author: elsat
"""
# projet de bac 3 : simulation numérique

import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.1
t_max = 5000
M = int(t_max/delta_t)+1
t = np.linspace(0,t_max, M)
G = 39.43461 #en ua^3/ms an^2
m_s = 1
m_j = 0.0009548
m_sat = 0.0002858
Q_s = np.zeros((M,3))
P_s = np.zeros ((M,3))
Q_j = np.zeros((M,3))
P_j = np.zeros ((M,3))
Q_j[0,:] = np.array([4.60945,1.70369,0.61805])
P_j[0,:] = np.array([m_j*(-1.04025), m_j*(2.46886), m_j*(1.08405)])
Q_sat = np.zeros((M,3))
P_sat = np.zeros ((M,3))
Q_sat[0,:] = np.array([8.44247,-4.47389,-2.21141])
P_sat[0,:] = np.array([m_sat*(0.9198), m_sat*(1.6352), m_sat*(0.6351)])
m_me = 1.6596*10**(-7)
Q_me = np.zeros((M,3))
P_me = np.zeros ((M,3))
Q_me[0,:] = np.array([-0.395067 , -0.09384 , -0.009179])
P_me[0,:] = np.array([m_me*(0.142058), m_me*(-8.466175), m_me*(-4.537315)])
m_v = 2.4485*10**(-6)
Q_v = np.zeros((M,3))
P_v = np.zeros ((M,3))
Q_v[0,:] = np.array([-0.55638 , 0.400395 ,0.215363])
P_v[0,:] = np.array([m_v*(-4.6894105), m_v*(-5.3631275), m_v*(-5.3631275)])
m_t = 3.0035*10**(-6)
Q_t = np.zeros((M,3))
P_t = np.zeros ((M,3))
Q_t[0,:] = np.array([-0.846289 , -0.497922 ,-0.21584 ])
P_t[0,:] = np.array([m_t*(3.290037), m_t*(4.872823), m_t*(-2.112558)])
m_ma = 3.2279*10**(-7)
Q_ma = np.zeros((M,3))
P_ma = np.zeros ((M,3))
Q_ma[0,:] = np.array([-1.265177 , 0.962068 ,0.475416 ])
P_ma[0,:] = np.array([m_ma*(-3.1091795), m_ma*(-3.17769), m_ma*(-1.3736775)])
m_u = 4.36532*10**(-5)
Q_u = np.zeros((M,3))
P_u = np.zeros ((M,3))
Q_u[0,:] = np.array([13.03868 , 13.53642 ,5.7440599 ])
P_u[0,:] = np.array([m_u*(-1.085875), m_u*(0.8081465), m_u*(0.3692705)])
m_n = 5.02917*10**(-5)
Q_n = np.zeros((M,3))
P_n = np.zeros ((M,3))
Q_n[0,:] = np.array([29.79099 , -2.12289 ,-1.610586 ])
P_n[0,:] = np.array([m_n*(0.0908412), m_n*(1.0666322), m_n*(0.4343573)])

m_sys = np.array([m_s,m_me,m_v,m_t,m_ma,m_j,m_sat,m_u,m_n])
Q_sys = np.array([Q_s,Q_me,Q_v,Q_t,Q_ma,Q_j,Q_sat,Q_u,Q_n])
P_sys = np.array([P_s,P_me,P_v,P_t,P_ma,P_j,P_sat,P_u,P_n])



def Heun_2_non_inertiel(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2):
    for i in range (M-1):
        Q_tilde_1 = np.zeros(3)
        Q_tilde_2 = np.zeros(3)
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        diff = Q_2[i] - Q_1[i]
        dist = np.linalg.norm(Q_2[i]-Q_1[i])


        Q_tilde_1 = Q_1[i] + (delta_t * (P_1[i]/m_1))
        P_tilde_1 = P_1[i] + (delta_t * G*m_1*m_2*(1/dist**3)*diff)
        Q_tilde_2 = Q_2[i] + (delta_t * (P_2[i]/m_2))
        P_tilde_2 = P_2[i] - (delta_t * G*m_1*m_2*(1/dist**3)*diff)
        
        diff_tilde = Q_tilde_2 - Q_tilde_1
        dist_tilde = np.linalg.norm(Q_tilde_2-Q_tilde_1)


        Q_1[i+1] = Q_1[i] + (delta_t/2 * ((P_1[i]/m_1) + (P_tilde_1/m_1)))
        P_1[i+1] = P_1[i] + (delta_t/2 * (G*m_1*m_2*(1/dist**3)*diff + G*m_1*m_2*(1/dist_tilde**3)*diff_tilde))
        Q_2[i+1] = Q_2[i] + (delta_t/2 * ((P_2[i]/m_2) + (P_tilde_2/m_2)))
        P_2[i+1] = P_2[i] - (delta_t/2 * (G*m_1*m_2*(1/dist**3)*diff + G*m_1*m_2*(1/dist_tilde**3)*diff_tilde))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[:,0],Q_1[:,1],Q_1[:,2], label = astre_1)
    ax.plot3D(Q_2[:,0],Q_2[:,1],Q_2[:,2], label = astre_2)
    plt.title(f"Trajectoires selon le schéma de Heun ΔT = {delta_t}")
    plt.legend()
    plt.show()


def Verlet_2_non_inertiel (Q_1,P_1,m_1, astre_1, Q_2 , P_2 , m_2, astre_2 ):
    for i in range (M-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        diff_1 = Q_2[i] - Q_1[i]
        dist_1 = np.linalg.norm(Q_2[i]-Q_1[i])
        
        
        P_tilde_1 = P_1[i] + (delta_t * G*m_1*m_2*(1/dist_1**3)*diff_1)
        P_tilde_2 = P_2[i] - (delta_t * G*m_1*m_2*(1/dist_1**3)*diff_1)


        Q_1[i+1] = Q_1[i] + (delta_t/(2*m_1) * (P_1[i] + P_tilde_1))
        Q_2[i+1] = Q_2[i] + (delta_t/(2*m_2) * (P_2[i] + P_tilde_2))
        
        diff_2 = Q_2[i+1] - Q_1[i+1]
        dist_2 = np.linalg.norm(Q_2[i+1]-Q_1[i+1])
        
        P_1[i+1] = P_1[i] + (delta_t/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
        P_2[i+1] = P_2[i] - (delta_t/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[:,0],Q_1[:,1],Q_1[:,2], label = astre_1)
    ax.plot3D(Q_2[:,0],Q_2[:,1],Q_2[:,2], label = astre_2)
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Verlet ΔT = {delta_t}")
    plt.show()



def Heun_2(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2):
    for i in range (M-1):
        Q_tilde_1 = np.zeros(3)
        Q_tilde_2 = np.zeros(3)
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        diff = Q_2[i] - Q_1[i]
        dist = np.linalg.norm(Q_2[i]-Q_1[i])


        Q_tilde_1 = Q_1[i] + (delta_t * (P_1[i]/m_1))
        P_tilde_1 = P_1[i] + (delta_t * G*m_1*m_2*(1/dist**3)*diff)
        Q_tilde_2 = Q_2[i] + (delta_t * (P_2[i]/m_2))
        P_tilde_2 = P_2[i] - (delta_t * G*m_1*m_2*(1/dist**3)*diff)
        
        diff_tilde = Q_tilde_2 - Q_tilde_1
        dist_tilde = np.linalg.norm(Q_tilde_2-Q_tilde_1)


        Q_1[i+1] = Q_1[i] + (delta_t/2 * ((P_1[i]/m_1) + (P_tilde_1/m_1)))
        P_1[i+1] = P_1[i] + (delta_t/2 * (G*m_1*m_2*(1/dist**3)*diff + G*m_1*m_2*(1/dist_tilde**3)*diff_tilde))
        Q_2[i+1] = Q_2[i] + (delta_t/2 * ((P_2[i]/m_2) + (P_tilde_2/m_2)))
        P_2[i+1] = P_2[i] - (delta_t/2 * (G*m_1*m_2*(1/dist**3)*diff + G*m_1*m_2*(1/dist_tilde**3)*diff_tilde))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = astre_1, color="red")
    ax.plot3D(Q_2[:,0]-Q_1[:,0],Q_2[:,1]-Q_1[:,1],Q_2[:,2]-Q_1[:,2], label = astre_2)
    plt.title(f"Trajectoires selon le schéma de Heun ΔT = {delta_t}")
    plt.legend()
    plt.show()


def Verlet_2 (Q_1,P_1,m_1, astre_1, Q_2 , P_2 , m_2, astre_2 ):
    for i in range (M-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        diff_1 = Q_2[i] - Q_1[i]
        dist_1 = np.linalg.norm(Q_2[i]-Q_1[i])
        
        
        P_tilde_1 = P_1[i] + (delta_t * G*m_1*m_2*(1/dist_1**3)*diff_1)
        P_tilde_2 = P_2[i] - (delta_t * G*m_1*m_2*(1/dist_1**3)*diff_1)


        Q_1[i+1] = Q_1[i] + (delta_t/(2*m_1) * (P_1[i] + P_tilde_1))
        Q_2[i+1] = Q_2[i] + (delta_t/(2*m_2) * (P_2[i] + P_tilde_2))
        
        diff_2 = Q_2[i+1] - Q_1[i+1]
        dist_2 = np.linalg.norm(Q_2[i+1]-Q_1[i+1])
        
        P_1[i+1] = P_1[i] + (delta_t/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
        P_2[i+1] = P_2[i] - (delta_t/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = astre_1, color = "red")
    ax.plot3D(Q_2[:,0]-Q_1[:,0],Q_2[:,1]-Q_1[:,1],Q_2[:,2]-Q_1[:,2], label = astre_2)
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Verlet ΔT = {delta_t}")
    plt.show()
    
    
def Heun_3_non_inertiel(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2 ,Q_3,P_3,m_3, astre_3):
    for i in range (M-1):
        Q_tilde_1 = np.zeros(3)
        Q_tilde_2 = np.zeros(3)
        Q_tilde_3 = np.zeros(3)
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        P_tilde_3 = np.zeros(3)
        diff_21 = Q_2[i] - Q_1[i]
        diff_12 = Q_1[i] - Q_2[i]
        diff_31 = Q_3[i] - Q_1[i]
        diff_13 = Q_1[i] - Q_3[i]
        diff_23 = Q_2[i] - Q_3[i]
        diff_32 = Q_3[i] - Q_2[i]
        dist_21 = np.linalg.norm(Q_2[i]-Q_1[i])
        dist_12 = np.linalg.norm(Q_1[i]-Q_2[i])
        dist_31 = np.linalg.norm(Q_3[i]-Q_1[i])
        dist_13 = np.linalg.norm(Q_1[i]-Q_3[i])
        dist_23 = np.linalg.norm(Q_2[i]-Q_3[i])
        dist_32 = np.linalg.norm(Q_3[i]-Q_2[i])


        Q_tilde_1 = Q_1[i] + (delta_t * (P_1[i]/m_1))
        P_tilde_1 = P_1[i] + (delta_t * (G*m_1*m_2*(1/dist_21**3)*diff_21 + G*m_1*m_3*(1/dist_31**3)*diff_31))
        Q_tilde_2 = Q_2[i] + (delta_t * (P_2[i]/m_2))
        P_tilde_2 = P_2[i] + (delta_t * (G*m_1*m_2*(1/dist_12**3)*diff_12 + G*m_2*m_3*(1/dist_32**3)*diff_32))
        Q_tilde_3 = Q_3[i] + (delta_t * (P_3[i]/m_3))
        P_tilde_3 = P_3[i] + (delta_t * (G*m_1*m_3*(1/dist_13**3)*diff_13 + G*m_2*m_3*(1/dist_23**3)*diff_23))
        
        
        diff_tilde_21 = Q_tilde_2 - Q_tilde_1
        diff_tilde_12 = Q_tilde_1 - Q_tilde_2
        diff_tilde_31 = Q_tilde_3 - Q_tilde_1
        diff_tilde_13 = Q_tilde_1 - Q_tilde_3
        diff_tilde_23 = Q_tilde_2 - Q_tilde_3
        diff_tilde_32 = Q_tilde_3 - Q_tilde_2
        dist_tilde_21 = np.linalg.norm(Q_tilde_2-Q_tilde_1)
        dist_tilde_12 = np.linalg.norm(Q_tilde_1-Q_tilde_2)
        dist_tilde_31 = np.linalg.norm(Q_tilde_3-Q_tilde_1)
        dist_tilde_13 = np.linalg.norm(Q_tilde_1-Q_tilde_3)
        dist_tilde_23 = np.linalg.norm(Q_tilde_2-Q_tilde_3)
        dist_tilde_32 = np.linalg.norm(Q_tilde_3-Q_tilde_2)


        Q_1[i+1] = Q_1[i] + (delta_t/2 * ((P_1[i]/m_1) + (P_tilde_1/m_1)))
        P_1[i+1] = P_1[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_21**3)*diff_21 + G*m_1*m_3*(1/dist_31**3)*diff_31) + (G*m_1*m_2*(1/dist_tilde_21**3)*diff_tilde_21 + G*m_1*m_3*(1/dist_tilde_31**3)*diff_tilde_31)))
        Q_2[i+1] = Q_2[i] + (delta_t/2 * ((P_2[i]/m_2) + (P_tilde_2/m_2)))
        P_2[i+1] = P_2[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_12**3)*diff_12 + G*m_2*m_3*(1/dist_32**3)*diff_32) + (G*m_1*m_2*(1/dist_tilde_12**3)*diff_tilde_12 + G*m_2*m_3*(1/dist_tilde_32**3)*diff_tilde_32)))
        Q_3[i+1] = Q_3[i] + (delta_t/2 * ((P_3[i]/m_3) + (P_tilde_3/m_3)))
        P_3[i+1] = P_3[i] + (delta_t/2 * ((G*m_1*m_3*(1/dist_13**3)*diff_13 + G*m_2*m_3*(1/dist_23**3)*diff_23) + (G*m_1*m_3*(1/dist_tilde_13**3)*diff_tilde_13 + G*m_2*m_3*(1/dist_tilde_23**3)*diff_tilde_23)))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[:,0],Q_1[:,1],Q_1[:,2], label = astre_1)
    ax.plot3D(Q_2[:,0],Q_2[:,1],Q_2[:,2], label = astre_2)
    ax.plot3D(Q_3[:,0],Q_3[:,1],Q_3[:,2], label = astre_3)
    plt.title(f"Trajectoires selon le schéma de Heun ΔT = {delta_t}")
    plt.legend()
    plt.show()


def Verlet_3_non_inertiel(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2 ,Q_3,P_3,m_3, astre_3):
    for i in range (M-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        P_tilde_3 = np.zeros(3)
        diff_21_a = Q_2[i] - Q_1[i]
        diff_12_a = Q_1[i] - Q_2[i]
        diff_31_a = Q_3[i] - Q_1[i]
        diff_13_a = Q_1[i] - Q_3[i]
        diff_23_a = Q_2[i] - Q_3[i]
        diff_32_a = Q_3[i] - Q_2[i]
        dist_21_a = np.linalg.norm(Q_2[i]-Q_1[i])
        dist_12_a = np.linalg.norm(Q_1[i]-Q_2[i])
        dist_31_a = np.linalg.norm(Q_3[i]-Q_1[i])
        dist_13_a = np.linalg.norm(Q_1[i]-Q_3[i])
        dist_23_a = np.linalg.norm(Q_2[i]-Q_3[i])
        dist_32_a = np.linalg.norm(Q_3[i]-Q_2[i])
        

        P_tilde_1 = P_1[i] + (delta_t * (G*m_1*m_2*(1/dist_21_a**3)*diff_21_a + G*m_1*m_3*(1/dist_31_a**3)*diff_31_a))
        P_tilde_2 = P_2[i] + (delta_t * (G*m_1*m_2*(1/dist_12_a**3)*diff_12_a + G*m_2*m_3*(1/dist_32_a**3)*diff_32_a))
        P_tilde_3 = P_3[i] + (delta_t * (G*m_1*m_3*(1/dist_13_a**3)*diff_13_a + G*m_2*m_3*(1/dist_23_a**3)*diff_23_a))
        

        Q_1[i+1] = Q_1[i] + (delta_t/(2*m_1) * (P_1[i] + P_tilde_1))
        Q_2[i+1] = Q_2[i] + (delta_t/(2*m_2) * (P_2[i] + P_tilde_2))
        Q_3[i+1] = Q_3[i] + (delta_t/(2*m_3) * (P_3[i] + P_tilde_3))
        
        diff_21_b = Q_2[i+1] - Q_1[i+1]
        diff_12_b = Q_1[i+1] - Q_2[i+1]
        diff_31_b = Q_3[i+1] - Q_1[i+1]
        diff_13_b = Q_1[i+1] - Q_3[i+1]
        diff_23_b = Q_2[i+1] - Q_3[i+1]
        diff_32_b = Q_3[i+1] - Q_2[i+1]
        dist_21_b = np.linalg.norm(Q_2[i+1]-Q_1[i+1])
        dist_12_b = np.linalg.norm(Q_1[i+1]-Q_2[i+1])
        dist_31_b = np.linalg.norm(Q_3[i+1]-Q_1[i+1])
        dist_13_b = np.linalg.norm(Q_1[i+1]-Q_3[i+1])
        dist_23_b = np.linalg.norm(Q_2[i+1]-Q_3[i+1])
        dist_32_b = np.linalg.norm(Q_3[i+1]-Q_2[i+1])
        
        P_1[i+1] = P_1[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_21_a**3)*diff_21_a + G*m_1*m_3*(1/dist_31_a**3)*diff_31_a) + (G*m_1*m_2*(1/dist_21_b**3)*diff_21_b + G*m_1*m_3*(1/dist_31_b**3)*diff_31_b)))
        P_2[i+1] = P_2[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_12_a**3)*diff_12_a + G*m_2*m_3*(1/dist_32_a**3)*diff_32_a) + (G*m_1*m_2*(1/dist_12_b**3)*diff_12_b + G*m_2*m_3*(1/dist_32_b**3)*diff_32_b)))
        P_3[i+1] = P_3[i] + (delta_t/2 * ((G*m_1*m_3*(1/dist_13_a**3)*diff_13_a + G*m_2*m_3*(1/dist_23_a**3)*diff_23_a) + (G*m_1*m_3*(1/dist_13_b**3)*diff_13_b + G*m_2*m_3*(1/dist_23_b**3)*diff_23_b)))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[:,0],Q_1[:,1],Q_1[:,2], label = astre_1)
    ax.plot3D(Q_2[:,0],Q_2[:,1],Q_2[:,2], label = astre_2)
    ax.plot3D(Q_3[:,0],Q_3[:,1],Q_3[:,2], label = astre_3)
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Verlet ΔT = {delta_t}")
    plt.show()



def Heun_3(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2 ,Q_3,P_3,m_3, astre_3):
    for i in range (M-1):
        Q_tilde_1 = np.zeros(3)
        Q_tilde_2 = np.zeros(3)
        Q_tilde_3 = np.zeros(3)
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        P_tilde_3 = np.zeros(3)
        diff_21 = Q_2[i] - Q_1[i]
        diff_12 = Q_1[i] - Q_2[i]
        diff_31 = Q_3[i] - Q_1[i]
        diff_13 = Q_1[i] - Q_3[i]
        diff_23 = Q_2[i] - Q_3[i]
        diff_32 = Q_3[i] - Q_2[i]
        dist_21 = np.linalg.norm(Q_2[i]-Q_1[i])
        dist_12 = np.linalg.norm(Q_1[i]-Q_2[i])
        dist_31 = np.linalg.norm(Q_3[i]-Q_1[i])
        dist_13 = np.linalg.norm(Q_1[i]-Q_3[i])
        dist_23 = np.linalg.norm(Q_2[i]-Q_3[i])
        dist_32 = np.linalg.norm(Q_3[i]-Q_2[i])


        Q_tilde_1 = Q_1[i] + (delta_t * (P_1[i]/m_1))
        P_tilde_1 = P_1[i] + (delta_t * (G*m_1*m_2*(1/dist_21**3)*diff_21 + G*m_1*m_3*(1/dist_31**3)*diff_31))
        Q_tilde_2 = Q_2[i] + (delta_t * (P_2[i]/m_2))
        P_tilde_2 = P_2[i] + (delta_t * (G*m_1*m_2*(1/dist_12**3)*diff_12 + G*m_2*m_3*(1/dist_32**3)*diff_32))
        Q_tilde_3 = Q_3[i] + (delta_t * (P_3[i]/m_3))
        P_tilde_3 = P_3[i] + (delta_t * (G*m_1*m_3*(1/dist_13**3)*diff_13 + G*m_2*m_3*(1/dist_23**3)*diff_23))
        
        
        diff_tilde_21 = Q_tilde_2 - Q_tilde_1
        diff_tilde_12 = Q_tilde_1 - Q_tilde_2
        diff_tilde_31 = Q_tilde_3 - Q_tilde_1
        diff_tilde_13 = Q_tilde_1 - Q_tilde_3
        diff_tilde_23 = Q_tilde_2 - Q_tilde_3
        diff_tilde_32 = Q_tilde_3 - Q_tilde_2
        dist_tilde_21 = np.linalg.norm(Q_tilde_2-Q_tilde_1)
        dist_tilde_12 = np.linalg.norm(Q_tilde_1-Q_tilde_2)
        dist_tilde_31 = np.linalg.norm(Q_tilde_3-Q_tilde_1)
        dist_tilde_13 = np.linalg.norm(Q_tilde_1-Q_tilde_3)
        dist_tilde_23 = np.linalg.norm(Q_tilde_2-Q_tilde_3)
        dist_tilde_32 = np.linalg.norm(Q_tilde_3-Q_tilde_2)


        Q_1[i+1] = Q_1[i] + (delta_t/2 * ((P_1[i]/m_1) + (P_tilde_1/m_1)))
        P_1[i+1] = P_1[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_21**3)*diff_21 + G*m_1*m_3*(1/dist_31**3)*diff_31) + (G*m_1*m_2*(1/dist_tilde_21**3)*diff_tilde_21 + G*m_1*m_3*(1/dist_tilde_31**3)*diff_tilde_31)))
        Q_2[i+1] = Q_2[i] + (delta_t/2 * ((P_2[i]/m_2) + (P_tilde_2/m_2)))
        P_2[i+1] = P_2[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_12**3)*diff_12 + G*m_2*m_3*(1/dist_32**3)*diff_32) + (G*m_1*m_2*(1/dist_tilde_12**3)*diff_tilde_12 + G*m_2*m_3*(1/dist_tilde_32**3)*diff_tilde_32)))
        Q_3[i+1] = Q_3[i] + (delta_t/2 * ((P_3[i]/m_3) + (P_tilde_3/m_3)))
        P_3[i+1] = P_3[i] + (delta_t/2 * ((G*m_1*m_3*(1/dist_13**3)*diff_13 + G*m_2*m_3*(1/dist_23**3)*diff_23) + (G*m_1*m_3*(1/dist_tilde_13**3)*diff_tilde_13 + G*m_2*m_3*(1/dist_tilde_23**3)*diff_tilde_23)))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = astre_1, color = "red")
    ax.plot3D(Q_2[:,0]-Q_1[:,0],Q_2[:,1]-Q_1[:,1],Q_2[:,2]-Q_1[:,2], label = astre_2)
    ax.plot3D(Q_3[:,0]-Q_1[:,0],Q_3[:,1]-Q_1[:,1],Q_3[:,2]-Q_1[:,2], label = astre_3)
    plt.title(f"Trajectoires selon le schéma de Heun ΔT = {delta_t}")
    plt.legend()
    plt.show()


def energie_heun(Q_1, P_1, m_1, Q_2, P_2, m_2, Q_3, P_3, m_3):
    
    E_cin_1 = np.linalg.norm(P_1, axis=1)**2/2*m_1
    E_cin_2 = np.linalg.norm(P_2, axis=1)**2/2*m_2
    E_cin_3 = np.linalg.norm(P_3, axis=1)**2/2*m_3
    
    E_pot_3 = -G*m_1*m_2*(1/np.linalg.norm(Q_1-Q_2, axis = 1))
    E_pot_2 = -G*m_1*m_3*(1/np.linalg.norm(Q_1-Q_3, axis = 1))
    E_pot_1 = -G*m_2*m_3*(1/np.linalg.norm(Q_2-Q_3, axis = 1))
    
    E = E_cin_1 + E_cin_2 + E_cin_3 + E_pot_3 + E_pot_2 + E_pot_1
    
    for i in range(3):
        plt.plot(t, E)
    plt.xlabel("temps")
    plt.ylabel("Energie")
    plt.title("Conservation de l'énergie totale selon le schéma de Heun")
    plt.show()
    
    
def moment_ang_heun(Q_1, P_1, Q_2, P_2, Q_3, P_3):
    L_1 = np.zeros((M,3))
    L_2 = np.zeros((M,3))
    L_3 = np.zeros((M,3))
    
    L_1[:,0] = Q_1[:,1]*P_1[:,2] - Q_1[:,2]*P_1[:,1]
    L_1[:,1] = Q_1[:,2]*P_1[:,0] - Q_1[:,0]*P_1[:,2]
    L_1[:,2] = Q_1[:,0]*P_1[:,1] - Q_1[:,1]*P_1[:,0]
    
    L_2[:,0] = Q_2[:,1]*P_2[:,2] - Q_2[:,2]*P_2[:,1]
    L_2[:,1] = Q_2[:,2]*P_2[:,0] - Q_2[:,0]*P_2[:,2]
    L_2[:,2] = Q_2[:,0]*P_2[:,1] - Q_2[:,1]*P_2[:,0]

    L_3[:,0] = Q_3[:,1]*P_3[:,2] - Q_3[:,2]*P_3[:,1]
    L_3[:,1] = Q_3[:,2]*P_3[:,0] - Q_3[:,0]*P_3[:,2]
    L_3[:,2] = Q_3[:,0]*P_3[:,1] - Q_3[:,1]*P_3[:,0]

    L = L_1 + L_2 + L_3
    
    for i in range(3):
        plt.plot(t, L[:,i], label="coordonnée "+str(i+1))
    plt.legend()
    plt.xlabel("temps")
    plt.ylabel("Moment angulaire")
    plt.title("Conservation du moment angulaire total selon le schéma de Heun")
    plt.show()


def Verlet_3(Q_1,P_1,m_1, astre_1, Q_2, P_2, m_2, astre_2 ,Q_3,P_3,m_3, astre_3):
    for i in range (M-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        P_tilde_3 = np.zeros(3)
        diff_21_a = Q_2[i] - Q_1[i]
        diff_12_a = Q_1[i] - Q_2[i]
        diff_31_a = Q_3[i] - Q_1[i]
        diff_13_a = Q_1[i] - Q_3[i]
        diff_23_a = Q_2[i] - Q_3[i]
        diff_32_a = Q_3[i] - Q_2[i]
        dist_21_a = np.linalg.norm(Q_2[i]-Q_1[i])
        dist_12_a = np.linalg.norm(Q_1[i]-Q_2[i])
        dist_31_a = np.linalg.norm(Q_3[i]-Q_1[i])
        dist_13_a = np.linalg.norm(Q_1[i]-Q_3[i])
        dist_23_a = np.linalg.norm(Q_2[i]-Q_3[i])
        dist_32_a = np.linalg.norm(Q_3[i]-Q_2[i])
        

        P_tilde_1 = P_1[i] + (delta_t * (G*m_1*m_2*(1/dist_21_a**3)*diff_21_a + G*m_1*m_3*(1/dist_31_a**3)*diff_31_a))
        P_tilde_2 = P_2[i] + (delta_t * (G*m_1*m_2*(1/dist_12_a**3)*diff_12_a + G*m_2*m_3*(1/dist_32_a**3)*diff_32_a))
        P_tilde_3 = P_3[i] + (delta_t * (G*m_1*m_3*(1/dist_13_a**3)*diff_13_a + G*m_2*m_3*(1/dist_23_a**3)*diff_23_a))
        

        Q_1[i+1] = Q_1[i] + (delta_t/(2*m_1) * (P_1[i] + P_tilde_1))
        Q_2[i+1] = Q_2[i] + (delta_t/(2*m_2) * (P_2[i] + P_tilde_2))
        Q_3[i+1] = Q_3[i] + (delta_t/(2*m_3) * (P_3[i] + P_tilde_3))
        
        diff_21_b = Q_2[i+1] - Q_1[i+1]
        diff_12_b = Q_1[i+1] - Q_2[i+1]
        diff_31_b = Q_3[i+1] - Q_1[i+1]
        diff_13_b = Q_1[i+1] - Q_3[i+1]
        diff_23_b = Q_2[i+1] - Q_3[i+1]
        diff_32_b = Q_3[i+1] - Q_2[i+1]
        dist_21_b = np.linalg.norm(Q_2[i+1]-Q_1[i+1])
        dist_12_b = np.linalg.norm(Q_1[i+1]-Q_2[i+1])
        dist_31_b = np.linalg.norm(Q_3[i+1]-Q_1[i+1])
        dist_13_b = np.linalg.norm(Q_1[i+1]-Q_3[i+1])
        dist_23_b = np.linalg.norm(Q_2[i+1]-Q_3[i+1])
        dist_32_b = np.linalg.norm(Q_3[i+1]-Q_2[i+1])
        
        P_1[i+1] = P_1[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_21_a**3)*diff_21_a + G*m_1*m_3*(1/dist_31_a**3)*diff_31_a) + (G*m_1*m_2*(1/dist_21_b**3)*diff_21_b + G*m_1*m_3*(1/dist_31_b**3)*diff_31_b)))
        P_2[i+1] = P_2[i] + (delta_t/2 * ((G*m_1*m_2*(1/dist_12_a**3)*diff_12_a + G*m_2*m_3*(1/dist_32_a**3)*diff_32_a) + (G*m_1*m_2*(1/dist_12_b**3)*diff_12_b + G*m_2*m_3*(1/dist_32_b**3)*diff_32_b)))
        P_3[i+1] = P_3[i] + (delta_t/2 * ((G*m_1*m_3*(1/dist_13_a**3)*diff_13_a + G*m_2*m_3*(1/dist_23_a**3)*diff_23_a) + (G*m_1*m_3*(1/dist_13_b**3)*diff_13_b + G*m_2*m_3*(1/dist_23_b**3)*diff_23_b)))
     
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = astre_1, color = "red")
    ax.plot3D(Q_2[:,0]-Q_1[:,0],Q_2[:,1]-Q_1[:,1],Q_2[:,2]-Q_1[:,2], label = astre_2)
    ax.plot3D(Q_3[:,0]-Q_1[:,0],Q_3[:,1]-Q_1[:,1],Q_3[:,2]-Q_1[:,2], label = astre_3)
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Verlet ΔT = {delta_t}")
    plt.show()


def energie_verlet(Q_1, P_1, m_1, Q_2, P_2, m_2, Q_3, P_3, m_3):
    
    E_cin_1 = np.linalg.norm(P_1, axis=1)**2/2*m_1
    E_cin_2 = np.linalg.norm(P_2, axis=1)**2/2*m_2
    E_cin_3 = np.linalg.norm(P_3, axis=1)**2/2*m_3
    
    E_pot_3 = -G*m_1*m_2*(1/np.linalg.norm(Q_1-Q_2, axis = 1))
    E_pot_2 = -G*m_1*m_3*(1/np.linalg.norm(Q_1-Q_3, axis = 1))
    E_pot_1 = -G*m_2*m_3*(1/np.linalg.norm(Q_2-Q_3, axis = 1))
    
    E = E_cin_1 + E_cin_2 + E_cin_3 + E_pot_3 + E_pot_2 + E_pot_1
    
    for i in range(3):
        plt.plot(t, E)
    plt.xlabel("temps")
    plt.ylabel("Energie")
    plt.title("Conservation de l'énergie totale selon le schéma de Verlet")
    plt.show()
    
def moment_ang_verlet(Q_1, P_1, Q_2, P_2, Q_3, P_3):
    L_1 = np.zeros((M,3))
    L_2 = np.zeros((M,3))
    L_3 = np.zeros((M,3))
    
    L_1[:,0] = Q_1[:,1]*P_1[:,2] - Q_1[:,2]*P_1[:,1]
    L_1[:,1] = Q_1[:,2]*P_1[:,0] - Q_1[:,0]*P_1[:,2]
    L_1[:,2] = Q_1[:,0]*P_1[:,1] - Q_1[:,1]*P_1[:,0]
    
    L_2[:,0] = Q_2[:,1]*P_2[:,2] - Q_2[:,2]*P_2[:,1]
    L_2[:,1] = Q_2[:,2]*P_2[:,0] - Q_2[:,0]*P_2[:,2]
    L_2[:,2] = Q_2[:,0]*P_2[:,1] - Q_2[:,1]*P_2[:,0]

    L_3[:,0] = Q_3[:,1]*P_3[:,2] - Q_3[:,2]*P_3[:,1]
    L_3[:,1] = Q_3[:,2]*P_3[:,0] - Q_3[:,0]*P_3[:,2]
    L_3[:,2] = Q_3[:,0]*P_3[:,1] - Q_3[:,1]*P_3[:,0]

    L = L_1 + L_2 + L_3
    
    for i in range(3):
        plt.plot(t, L[:,i], label="coordonnée "+str(i+1))
    plt.legend()
    plt.xlabel("temps")
    plt.ylabel("Moment angulaire")
    plt.title("Conservation du moment angulaire total selon le schéma de Verlet")
    plt.show()


def Heun_9 (Q,P,m) :
    
    for i in range(M-1):
        Q_tilde = np.zeros((9,3))
        P_tilde = np.zeros((9,3))
        
        for k in range(9): 
            Q_tilde[k] = Q[k][i] + (delta_t * (P[k][i]/m[k]))
            F = np.array([0,0,0])
            for l in range(9):
                if k!=l:
                    diff = Q[l][i] - Q[k][i]
                    dist = np.linalg.norm(diff)
                    F = F + G*m[k]*m[l]*(1/dist**3)*diff
            P_tilde[k] = P[k][i] + delta_t * F
            
        for k in range(9):                
            Q[k][i+1] = Q[k][i] + delta_t/2 * (P[k][i]/m[k] + P_tilde[k]/m[k])
            F = np.array([0,0,0])
            for l in range(9):
                if k!=l:
                    diff = Q[l][i] - Q[k][i]
                    dist = np.linalg.norm(diff)
                    diff_tilde = Q_tilde[l] - Q_tilde[k]
                    dist_tilde = np.linalg.norm(diff_tilde)
                    F = F + G*m[k]*m[l]*( (1/dist**3)*diff + (1/dist_tilde**3)*diff_tilde)
            P[k][i+1] = P[k][i] + delta_t/2 * F
            
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = "Soleil")
    ax.plot3D(Q[1][:,0]-Q[0][:,0],Q[1][:,1]-Q[0][:,1],Q[1][:,2]-Q[0][:,2], label = "Mercure")
    ax.plot3D(Q[2][:,0]-Q[0][:,0],Q[2][:,1]-Q[0][:,1],Q[2][:,2]-Q[0][:,2], label = "Vénus")
    ax.plot3D(Q[3][:,0]-Q[0][:,0],Q[3][:,1]-Q[0][:,1],Q[3][:,2]-Q[0][:,2], label = "Terre")
    ax.plot3D(Q[4][:,0]-Q[0][:,0],Q[4][ :,1]-Q[0][:,1],Q[4][:,2]-Q[0][:,2], label = "Mars")
    ax.plot3D(Q[5][:,0]-Q[0][:,0],Q[5][:,1]-Q[0][:,1],Q[5][:,2]-Q[0][:,2], label = "Jupiter")
    ax.plot3D(Q[6][:,0]-Q[0][:,0],Q[6][:,1]-Q[0][:,1],Q[6][:,2]-Q[0][:,2], label ="Saturne")
    ax.plot3D(Q[7][:,0]-Q[0][:,0],Q[7][:,1]-Q[0][:,1],Q[7][:,2]-Q[0][:,2], label = "Uranus")
    ax.plot3D(Q[8][:,0]-Q[0][:,0],Q[8][:,1]-Q[0][:,1],Q[8][:,2]-Q[0][:,2], label = "Neptune")
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Heun ΔT = {delta_t}")
    plt.show()


def Verlet_9(Q,P,m):
    
    for i in range (M-1):
        P_tilde = np.zeros((9,3))
       
        for k in range(9):
            F = np.array([0,0,0])
            for l in range (9):
                if k!=l:
                    diff_1 = Q[l][i] - Q[k][i]
                    dist_1 = np.linalg.norm(diff_1)
                    F = F + G*m[k]*m[l]*(1/dist_1**3)*diff_1
            P_tilde[k] = P[k][i] + delta_t * F
            
        for k in range(9):
            Q[k][i+1] = Q[k][i] + delta_t/(2*m[k]) * (P[k][i] + P_tilde[k])
            F = np.array([0,0,0])   
            for l in range (9):
                if k!=l:
                    diff_1 = Q[l][i] - Q[k][i]
                    dist_1 = np.linalg.norm(diff_1)
                    diff_2 = Q[l][i+1] - Q[k][i+1]
                    dist_2 = np.linalg.norm(diff_2)
                    F = F + G*m[k]*m[l]*( (1/dist_1**3)*diff_1 + (1/dist_2**3)*diff_2)
            P[k][i+1] = P[k][i] + delta_t/2 * F
        
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0], label = "Soleil")
    ax.plot3D(Q[1][:,0]-Q[0][:,0],Q[1][:,1]-Q[0][:,1],Q[1][:,2]-Q[0][:,2], label = "Mercure")
    ax.plot3D(Q[2][:,0]-Q[0][:,0],Q[2][:,1]-Q[0][:,1],Q[2][:,2]-Q[0][:,2], label = "Vénus")
    ax.plot3D(Q[3][:,0]-Q[0][:,0],Q[3][:,1]-Q[0][:,1],Q[3][:,2]-Q[0][:,2], label = "Terre")
    ax.plot3D(Q[4][:,0]-Q[0][:,0],Q[4][ :,1]-Q[0][:,1],Q[4][:,2]-Q[0][:,2], label = "Mars")
    ax.plot3D(Q[5][:,0]-Q[0][:,0],Q[5][:,1]-Q[0][:,1],Q[5][:,2]-Q[0][:,2], label = "Jupiter")
    ax.plot3D(Q[6][:,0]-Q[0][:,0],Q[6][:,1]-Q[0][:,1],Q[6][:,2]-Q[0][:,2], label ="Saturne")
    ax.plot3D(Q[7][:,0]-Q[0][:,0],Q[7][:,1]-Q[0][:,1],Q[7][:,2]-Q[0][:,2], label = "Uranus")
    ax.plot3D(Q[8][:,0]-Q[0][:,0],Q[8][:,1]-Q[0][:,1],Q[8][:,2]-Q[0][:,2], label = "Neptune")
    plt.legend()
    plt.title(f"Trajectoires selon le schéma de Verlet ΔT = {delta_t}")
    plt.show()



#Heun_2_non_inertiel (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter")
#Verlet_2_non_inertiel (Q_s, P_s, m_s,"Soleil", Q_j, P_j, m_j, "Jupiter")
#Heun_2 (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter")
#Verlet_2 (Q_s, P_s, m_s,"Soleil", Q_j, P_j, m_j, "Jupiter")
#Heun_3_non_inertiel (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter", Q_sat, P_sat, m_sat, "Saturne")
#Verlet_3_non_inertiel (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter", Q_sat, P_sat, m_sat, "Saturne")
#Heun_3 (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter", Q_sat, P_sat, m_sat, "Saturne")
#energie_heun(Q_s , P_s, m_s, Q_j, P_j, m_j, Q_sat, P_sat, m_sat)
#moment_ang_heun(Q_s, P_s, Q_j,P_j, Q_sat, P_sat)
Verlet_3 (Q_s, P_s, m_s, "Soleil", Q_j, P_j, m_j, "Jupiter", Q_sat, P_sat, m_sat, "Saturne")
energie_verlet(Q_s , P_s, m_s, Q_j, P_j, m_j, Q_sat, P_sat, m_sat)
moment_ang_verlet(Q_s, P_s, Q_j,P_j, Q_sat, P_sat)
#Heun_9 (Q_sys, P_sys, m_sys)
#Verlet_9 (Q_sys, P_sys, m_sys)
