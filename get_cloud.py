# import dependencies
import pandas as pd
import csv
import numpy as np
import open3d as o3d
import plotly.graph_objects as go


# # read csv
# df = pd.read_csv('values2.csv')
# df.head()

# # test frame
# my_frame = 'final2849.jpg'

# image = df.loc[df['frame'] == my_frame]
# quats = image[['qw','qx','qy','qz']]
# quaternions = quats.to_numpy()
# print(quaternions)

# trans = image[['tx','ty','tz']]
# translation = trans.to_numpy()
# print(translation)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


def display_pointcloud(markers):

    labels = ["Carangidae", "Dinolestidae", "Enoplosidae", "Girellidae", "Microcanthidae", "Plesiopidae"]
    print(markers)

    # # test frame
    # my_frame = 'final2849.jpg'

    # image = df.loc[df['frame'] == my_frame]
    # quats = image[['qw','qx','qy','qz']]
    # quaternions = quats.to_numpy()
    # print(quaternions)

    # trans = image[['tx','ty','tz']]
    # translation = trans.to_numpy()
    # print(translation)

    # # get camera position
    # rt = quaternion_rotation_matrix(quaternions.transpose())
    # print(rt)

    # coords = np.matmul(rt.transpose(),translation.transpose())
    # print(coords)

    # read cloud
    cloud = o3d.io.read_point_cloud("cloud2.ply")

    points = np.asarray(cloud.points)

    colors = None
    if cloud.has_colors():
        colors = np.asarray(cloud.colors)
    elif cloud.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(cloud.normals) * 0.5
    else:
        geometry.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(geometry.colors)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:,0], y=points[:,1], z=points[:,2]*-1, 
                mode='markers',
                marker=dict(size=1, color=colors)
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )

    # camx=np.array([1,1,1]) * coords[0,0]
    # camy=np.array([1,1,1]) * coords[0,1]
    # camz=np.array([-1,-1,-1]) * coords[0,2]

    # fig.add_trace(go.Scatter3d(x=camx,y=camy,z=camz,
    #                                 mode='markers'))
    fig.show()