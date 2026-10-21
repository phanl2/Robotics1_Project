import numpy as np
from general_robotics_toolbox import subproblem1, subproblem3, subproblem4

# Rotation helper
def rot(k, q):
    """Rotate around axis k by q radians."""
    k = np.asarray(k, dtype=float).reshape(3)
    k = k / np.linalg.norm(k)

    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)

# Dofbot inverse kinematics
def invkin_subproblems_Dofbot(Rot, Pot):
    Rot = np.asarray(Rot, dtype=float).reshape(3, 3)
    Pot = np.asarray(Pot, dtype=float).reshape(3)

    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # Link lengths (meters)
    l0 = 0.061
    l1 = 0.0435
    l2 = 0.08285
    l3 = 0.08285
    l4 = 0.07385
    l5 = 0.05457

    L1 = l0 + l1
    L4 = l4 + l5

    q = np.full((5, 4), np.nan)

    # Find theta = q2 + q3 + q4
    d = np.dot(ez, Rot @ ex)
    theta_solutions = subproblem4(ez, ex, -ey, d)

    if len(theta_solutions) == 0:
        return np.empty((5, 0))

    if len(theta_solutions) == 1:
        theta = np.array([
            theta_solutions[0], np.nan,
            theta_solutions[0], np.nan
        ])
    else:
        theta = np.array([
            theta_solutions[0], theta_solutions[1],
            theta_solutions[0], theta_solutions[1]
        ])

    # Find q1 (subproblem1)
    for ii in range(4):
        if np.isnan(theta[ii]):
            continue

        p1 = rot(-ey, theta[ii]) @ ex
        p2 = Rot @ ex

        try:
            q[0, ii] = subproblem1(p1, p2, ez)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

    # Find q5 (subproblem1)
    for ii in range(4):
        if np.isnan(theta[ii]) or np.isnan(q[0, ii]):
            continue

        p1 = rot(ey, theta[ii]) @ ez
        p2 = Rot.T @ ez

        try:
            q[4, ii] = subproblem1(p1, p2, ex)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

    # Find q3 (subproblem3)
    for ii in range(2):
        if np.isnan(theta[ii]) or np.isnan(q[0, ii]):
            continue

        Pprime = (
            rot(ez, -q[0, ii]) @ (Pot - L1 * ez)
            + rot(ey, -theta[ii]) @ (L4 * ex)
        )

        p1 = l3 * ez
        p2 = l2 * ex
        d = np.linalg.norm(Pprime)

        try:
            q3_solutions = subproblem3(p1, -p2, -ey, d)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

        if len(q3_solutions) == 1:
            q[2, ii] = q3_solutions[0]
        elif len(q3_solutions) >= 2:
            q[2, ii] = q3_solutions[0]
            q[2, ii + 2] = q3_solutions[1]

    # Find q2 (subproblem1)
    for ii in range(4):
        if np.isnan(q[2, ii]) or np.isnan(q[0, ii]):
            continue

        Pprime = (
            rot(ez, -q[0, ii]) @ (Pot - L1 * ez)
            + rot(ey, -theta[ii]) @ (L4 * ex)
        )

        p1 = l2 * ex - rot(ey, -q[2, ii]) @ (l3 * ez)

        try:
            q[1, ii] = subproblem1(p1, Pprime, -ey)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

    # Find q4
    for ii in range(4):
        if not np.isnan(q[2, ii]) and not np.isnan(q[1, ii]):
            q[3, ii] = theta[ii] - q[1, ii] - q[2, ii]

    # Remove invalid solutions
    q = q[:, ~np.any(np.isnan(q), axis=0)]

    # Convert radians to degrees
    q = np.rad2deg(q)

    # MATLAB angle adjustment
    q[q < -180] += 360
    q[q > 360] -= 360

    return q


# INPUT VALUES
if __name__ == "__main__":

    # Desired end-effector orientation
    Rot = np.array([[0., -0.,  -1.],
                    [-0., -1.,  0.],
                    [-1.,  0., 0.],
                    ])

    # Desired end-effector position (x, y, z), meters (round to 4th decimal place)
    Pot = np.array([0., 0., 0.3986])

    # RUN INVERSE KINEMATICS
    print("Inverse Kinematic:")

    print("Input ROT matrix:")
    print(np.round(Rot, 4))

    print("\nInput POT (meters):")
    print(np.round(Pot, 4))

    try:
        q = invkin_subproblems_Dofbot(Rot, Pot)

        if q.shape[1] == 0:
            print("\nNo valid IK solutions found.")
        else:
            print("\nJoint angles (degrees), each column is one solution:")
            print(np.round(q, 4))

    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        print("\nIK calculation failed:", e)
