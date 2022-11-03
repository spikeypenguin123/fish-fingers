# q2e.py convert from quaternions to euler
import matplotlib as ply 
import numpy as np 




def q2e(q):
    # input:  vertical array of quaternions in the form [qw; qx; qy; qz] = [q0; q1; q2; q3]
    # output: vertical array of euler coordinates in the form [roll; pitch; yaw]
    
    eulers = np.zeros((len(q),len(q[0])))
    for i in range(1,len(q)):

        q_0 = q[i-1][0]
        q_1 = q[i-1][1]
        q_2 = q[i-1][2]
        q_3 = q[i-1][3]

        # radians
        eulers[i-1][0] = [np.arctan2(2*(q_0*q_1+q_2*q_3),1-2*(q_1^2+q_2^2))] 
        eulers[i-1][1] = [np.arcsin(2*(q_0*q_2-q_3*q_1))]
        eulers[i-1][2] = [np.arctan2(2*(q_0*q_3+q_1*q_2),1-2*(q_2^2+q_3^2))]

        # degrees
        # * 180 / np.pi

    return eulers 


if __name__ == "__main__":
    q = [[1,0,0,0],[1,1,0,0]]
    e_verif = [[0,0,0],[116.56505117707799,0,0]]
    
    euler = q2e(q)

    for i in range(1,len(e_verif[:,0])):
        print('**************************************')
        print('Expected: ', e_verif[i-1,:])
        print('Result:   ', euler[:,i-1])