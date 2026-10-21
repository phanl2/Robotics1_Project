import numpy as np
from scipy.spatial.transform import Rotation

def rotx(q):
   q = np.radians(q)
   return np.array([[1, 0, 0],
                    [0, np.cos(q), -np.sin(q)],
                    [0, np.sin(q), np.cos(q)]])

def roty(q):
   q = np.radians(q)
   return np.array([[np.cos(q), 0, np.sin(q)],
                    [0, 1, 0],
                    [-np.sin(q), 0, np.cos(q)]])

def rotz(q):
   q = np.radians(q)
   return np.array([[np.cos(q), -np.sin(q), 0],
                    [np.sin(q), np.cos(q), 0],
                    [0, 0 , 1]])


def fk_Dofbot(q):
#    Computes end effector position and orientation relative to the base frame
#    Input: q: 5x1 number array of joint angles in degrees
#    Output: 3x1 numpy array describing position of end effector relative to base frame

    #q = np.array(q).flatten() #handles (5,1) shape input

    # Set up the basis unit vectors
    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    ez = np.array([0,0,1])

    # Define the link lengths in meters
    l0 = 0.061; # base to servo 1
    l1 = 0.0435; # servo 1 to servo 2
    l2 = 0.08285; # servo 2 to servo 3
    l3 = 0.08285; # servo 3 to servo 4
    l4 = 0.07385; # servo 4 to servo 5
    l5 = 0.05457; # servo 5 to gripper

    # set up the rotation matrices between subsequent frames

    R01 = rotz (q[0]); # rotation between base frame and 1 frame
    R12 = roty (-q[1]); # rotation between 1 and 2 frames
    R23 = roty (-q[2]); # rotation between 2 and 3 frames
    R34 = roty (-q[3]); # rotation between 3 and 4 frames
    R45 = rotx (-q[4]); # rotation between 4 and 5 frames
    R5T = np.eye(3)

    # set up the position vectors between subsequent frames
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -l3 * ez
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * ex

    # calculate Rot and Pot
    Rot = R01 @ R12 @ R23 @ R34 @ R45 @ R5T #matrix multiplication 

    Pot = P01 + R01 @ (P12 + R12 @ (P23 + R23 @ (P34 + R34 @(P45 + R45 @ P5T))))

    ROT = Rotation.from_matrix(Rot)
    POT = Pot.reshape(3, 1)

    return ROT, POT

if __name__ == "__main__":
    q = np.array([90, 90, 90, 90, 90])
    
    R0T, P0T = fk_Dofbot(q)

    print("R0T =")
    print(np.round(R0T.as_matrix(), 10))

    print("P0T =")
    print(np.round(P0T, 10))