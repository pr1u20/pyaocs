import numpy as np
import math

def wait_for_actuator(state, action):
    """Wait for the actuator to reach the desired state before firing."""

    if not np.array_equal((state[-4:]).astype(float), action[-4:].astype(float)):
        action[:2] = 0
        state[:2] = 0

        #print(f"[INFO] Waiting for actuator to reach target state: {action[-4:]}")
        
    return action, state

def Quaternion_to_Euler321(q):
    """Convert quaternion to euler in order 321 (ZYX) and in the fixed body reference frame."""

    x,y,z,w =q

    φ = math.atan2(2*((w*x)+(y*z)),1-2*((x**2)+y**2))

    θ = -(math.pi/2) +2*math.atan2(math.sqrt(1+2*((w*y)-(x*z))),math.sqrt(1-2*((w*y)-(x*z))))

    ψ = math.atan2(2*((w*z)+(x*y)),1-2*((y**2)+z**2))

    φ_deg = φ *180/math.pi
    θ_deg = θ *180/math.pi
    ψ_deg = ψ *180/math.pi

    return [ψ_deg, θ_deg, φ_deg]

def normalize_quaternion(q):
    """
    Normalize a quaternion to have unit length.

    Args:
        q: Quaternion as a numpy array [w, x, y, z].

    Returns:
        normalized_q: Normalized quaternion as a numpy array [w, x, y, z].
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        return q  # Avoid division by zero
    normalized_q = q / norm

    return normalized_q

def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to rotation matrix.

    :param quaternion: q (x, y, z, w)
    :type quaternion: list
    :return: R, rotation matrix
    :rtype: np.ndarray
    """

    x, y, z, w = quaternion  # Assuming quaternion as [a, b, c, d]

    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return R

def transform_vector(vector_world, orientation, inverse = False, euler = False):

    """
    If inverse == False, it can be used to transform a vector by the given orientation.

    If inverse == True it could be used for the follwoing:
    Convert vector from world reference frame to satellite reference frame. 
    vector_world can refer to the vector connecting the current position of the 
    satelite and the target position. The output would be that same vector in
    the satellite reference frame.

    :param vector_world: vector to transform.
    :type vector_world: list
    :param orientation: orientation of the satellite q (x, y, z, w).
    :type orientation: list
    :return: transformed vector
    :rtype: list
    """

    if euler == True:
        orientation = euler_to_quaternion(orientation)

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(orientation)

    # Convert the vector to a column vector
    vector_world = np.array(vector_world).reshape(-1, 1)

    # Apply inverse rotation to the vector
    if inverse == True:
        vector_satellite = np.dot(np.linalg.inv(rotation_matrix), vector_world)
    else:
        vector_satellite = np.dot(rotation_matrix, vector_world)

    return vector_satellite.flatten()  # Return as a flattened array

def euler_to_quaternion(euler):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """

    roll, pitch, yaw = euler


    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

class ExponentialMovingAverageFilter:
    def __init__(self, alpha=0.5):
        """
        Initializes the filter.
        
        :param alpha: Smoothing factor of the filter, 0 < alpha <= 1. 
                      A higher alpha discounts older observations faster.
        """
        self.alpha = alpha
        self.estimated_value = None

    def filter(self, new_value):
        """
        Applies the Exponential Moving Average filter to the new value.
        
        :param new_value: The new data point to be filtered.
        :return: The filtered value.
        """
        if self.estimated_value is None:
            # Initialize the estimated value with the first data point
            self.estimated_value = new_value
        else:
            # Apply the EMA formula
            self.estimated_value = self.alpha * new_value + (1 - self.alpha) * self.estimated_value
        
        return self.estimated_value