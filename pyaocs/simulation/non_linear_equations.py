import numpy as np
import time

from pyaocs.utils import normalize_quaternion, transform_vector
from pyaocs import parameters as param
from pyaocs.simulation.disturbances import Disturbances

def quaternion_derivative(q, w):
     """
     Calculate the derivative of a quaternion with respect to angular velocity.

     Args:
          q: Quaternion (numpy array) [x, y, z, w]
          w: Angular velocity (numpy array) [wx, wy, wz]

     Returns:
          dq_dt: Derivative of the quaternion with respect to angular velocity (numpy array) [dq/dt]
     """
     wx, wy, wz = w
     
     A = np.array([[0, wz, -wy, wx],
                    [-wz, 0, wx, wy],
                    [wy, -wx, 0, wz],
                    [-wx, -wy, -wz, 0]])

     dq_dt = 0.5 * np.dot(A, q)

     return dq_dt

def angular_velocity_derivative(I, w, Tc, Td):
     """Function to calculate the change in angular velocity given
     I, w, Tc and Td

     :param I: moment of inertia
     :type I: np.ndarray
     :param w: angular velocity
     :type w: np.ndarray
     :param Tc: control torque.
     :type Tc: np.ndarray
     :param Td: disturbance torque.
     :type Td: np.ndarray
     :return: change in angular velocity
     :rtype: np.ndarray
     """

     Ix, Iy, Iz = I
     wx, wy, wz = w

     H = np.array([(Iy - Iz)*wy*wz,
                    (Iz - Ix)*wx*wz,
                    (Ix - Iy)*wx*wy])
     
     dw_dt = np.divide((H + Tc + Td), I)

     return dw_dt

def force_calculation(inputs):

     F1, F2, γ1, γ2, δ1, δ2 = inputs

     θ_r1 = np.sqrt(γ1**2 + δ1**2)
     θ_r2 = np.sqrt(γ2**2 + δ2**2)

     # Find acceleration in every direction
     F = np.array([(F1*np.cos(θ_r1) - F2*np.cos(θ_r2)),
                   (F1*np.sin(δ1) - F2*np.sin(δ2)),
                   (- F1*np.sin(γ1) - F2*np.sin(γ2))
                   ])

     return F

class NonLinearPropagator():

     def __init__(self, dt, y0, use_disturbances = False):

          self.use_disturbances = use_disturbances

          self.dt = dt

          if use_disturbances:
               self.disturbance = Disturbances(self.dt)

          self.m = param.m # 1kg
          self.L = param.L # m
          self.I = param.I

          x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz = y0

          self.p = np.array([x, y, z])
          self.v = np.array([vx, vy, vz])
          self.q = np.array([q1, q2, q3, q4])
          self.w = np.array([wx, wy, wz])

          self.disturbances_F = []
          self.disturbances_T = []

          self.step = 0

     def rotation_derivative(self, inputs):

          F1, F2, γ1, γ2, δ1, δ2 = inputs

          self.q = normalize_quaternion(self.q)

          dq_dt = quaternion_derivative(self.q, self.w)

          Tx = 0
          Ty = -F1*self.L*np.sin(γ1) + F2*self.L*np.sin(γ2)
          Tz = -F1*self.L*np.sin(δ1) - F2*self.L*np.sin(δ2)

          Tc = np.array([Tx, Ty, Tz])

          if self.use_disturbances:
               Td = self.disturbance.calculate_disturbance_torque(self.p, self.q, self.step)
          else:
               Td = np.array([0, 0, 0])
          
          self.disturbances_T.append(Td)

          dw_dt = angular_velocity_derivative(self.I, self.w, Tc, Td)

          return dq_dt, dw_dt
     
     def position_derivative(self, inputs):
           
          Fc = force_calculation(inputs)

          if self.use_disturbances:
               Fd = self.disturbance.calculate_disturbance_force(self.p, self.q, self.step)
          else:
               Fd = np.array([0, 0, 0])

          self.disturbances_F.append(Fd)

          # Find acceleration in every direction
          F = Fc + Fd
          accel = F / self.m

          #Convert from body fixed to world inertial reference frame
          accel = transform_vector(accel, self.q)

          return accel

     def stepSimulation(self, inputs):

          # Perform rotattion propagation
          dq_dt, dw_dt = self.rotation_derivative(inputs)

          self.q += self.dt*dq_dt
          self.w += self.dt*dw_dt

          #Perform translation propagation 
          accel = self.position_derivative(inputs)

          #previous_v = self.v
          self.v += self.dt*accel
          #assuming that velocity chnages linearly, which is probably better aproximation than instantaneous change.
          #avg_v = (previous_v + self.v) / 2
          self.p += self.dt*self.v

          self.step += 1

          return self.p, self.v, self.q, self.w


def test_quaternion_derivative():

     # Example usage
     q = np.array([1.0, 0.0, 0.0, 0.0])  # Example quaternion [x, y, z, w]
     w = np.array([0.1, 0.2, 0.3])     # Example angular velocity [wx, wy, wz]

     dq_dt = quaternion_derivative(q, w)
     print("Derivative of the quaternion:", dq_dt)


if __name__ == "__main__":

     t = np.linspace(0, 10, 100)

     p = np.array(param.start_position)  # Example position [x, y, z]
     v = np.array(param.start_velocity)     # Example velocity [vx, vy, vz]
     q = np.array(param.start_orientation)  # Example quaternion [x, y, z, w]
     w = np.array(param.start_angular_velocity)     # Example angular velocity [wx, wy, wz]

     y0 = np.concatenate((p, v, q, w))
     dt = param.dt

     nl = NonLinearPropagator(dt = dt, y0=y0)

     F1 = 0
     F2 = 1
     γ1 = 0
     γ2 = 0
     δ1 = 0
     δ2 = param.thrust_vector_range * np.pi / 180

     inputs = F1, F2, γ1, γ2, δ1, δ2

     for _ in range(int(param.simulation_time / dt)):

          p, v, q, w = nl.stepSimulation(inputs)

          #time.sleep(dt)