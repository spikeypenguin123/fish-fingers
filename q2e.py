# q2e.py convert from quaternions to euler
import matplotlib as ply 
import numpy as np 


def q2e(q):
    # input:  vertical array of quaternions 
    # output: vertical array of euler coordinates 

    q_0 = q[1,:]
    q_1 = q[2,:]
    q_2 = q[3,:]
    q_3 = q[4,:]
    
    eulers = []
    for i in range(1,len(q_0)):
        eulers[:,i] = [[np.arctan2(2*(q_0*q_1+q_2*q_3),1-2*(q_1^2+q_2^2))], [np.arcsin(2*(q_0*q_2-q_3*q_1))], [np.arctan2(2*(q_0*q_3+q_1*q_2),1-2*(q_2^2+q_3^2))]]

    return eulers 
