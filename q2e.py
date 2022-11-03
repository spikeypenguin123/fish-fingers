# q2e.py convert from quaternions to euler
import matplotlib as ply 
import numpy as np 




def q2e(q):
    # input:  vertical array of quaternions in the form [qw; qx; qy; qz] = [q0; q1; q2; q3]
    # output: vertical array of euler coordinates in the form [roll; pitch; yaw]
    
    eulers = np.zeros((len(q),len(q[0])-1))
    for i in range(len(q)):

        # q_0 = q[i][0]
        # q_1 = q[i][1]
        # q_2 = q[i][2]
        # q_3 = q[i][3]
        q_0, q_1, q_2, q_3 = q[i]
        print('----------------------------------')
        print(q[i])
        print('----------------------------------')

        # radians
        eulers[i][0] = np.arctan2(2*(q_0*q_1+q_2*q_3),1-2*(q_1**2+q_2**2)) 
        eulers[i][1] = np.arcsin(2*(q_0*q_2-q_3*q_1))
        eulers[i][2] = np.arctan2(2*(q_0*q_3+q_1*q_2),1-2*(q_2**2+q_3**2))

        # degrees
        # * 180 / np.pi

    return eulers 


if __name__ == "__main__":
    q = [[1,0,0,0],[1,1,0,0],[0.97547004, 0.02369198, 0.21203752, -0.05419399]]
    e_verif = [[0,0,0],[2.034443935790656,0,0],[0,0,0,0]]
    # print(e_verif[0])
    euler = q2e(q)
    # print(euler)
    for i in range(len(e_verif)):
        print('**************************************')
        print('i: ', i)
        print('Expected: ', e_verif[i])
        print('Result:   ', euler[i])