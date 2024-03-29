import numpy as np
import random
import matplotlib.pyplot as plt

from pyaocs import parameters as param


class ESA_table():
    def __init__():
        pass
    def calculate_force(self):
        pass
    def calculate_torque(self):
        pass

class RandomDistubances():

    def __init__(self):
        pass

    def calculate_force(self):

        max_F = 0.05 # N

        x_min = - max_F
        x_max = max_F
        y_min = - max_F
        y_max = max_F
        z_min = 0
        z_max = 0

        Fx = random.uniform(x_min, x_max)
        Fy = random.uniform(y_min, y_max)
        Fz = random.uniform(z_min, z_max)

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self):

        max_T = 0.05 # Nm

        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = - max_T
        z_max = max_T

        Tx = random.uniform(x_min, x_max)
        Ty = random.uniform(y_min, y_max)
        Tz = random.uniform(z_min, z_max)

        Td = np.array([Tx, Ty, Tz])

        return Td

class SinusodialDistubances():

    def __init__(self):
        pass

    def calculate_force(self, t):

        period = 20

        max_F = 0.05 # N

        A_x = max_F
        A_y = max_F
        A_z = 0

        period_steps = period / param.dt
        Fx = A_x*np.sin(((2*np.pi) / period_steps) * t)
        Fy = A_y*np.sin(((2*np.pi) / period_steps) * t)
        Fz = A_z*np.sin(((2*np.pi) / period_steps) * t)

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self, t):

        period = 20

        max_T = 0.05 # N

        A_x = 0
        A_y = 0
        A_z = max_T

        period_steps = period / param.dt
        Tx = A_x*np.sin(((2*np.pi) / period_steps) * t)
        Ty = A_y*np.sin(((2*np.pi) / period_steps) * t)
        Tz = A_z*np.sin(((2*np.pi) / period_steps) * t)

        Td = np.array([Tx, Ty, Tz])

        return Td
    

class RandomSineNoise():

    def __init__(self, dt):

        self.dt = dt

        self.t = np.linspace(0, int(param.simulation_time / self.dt)-1, int(param.simulation_time / self.dt))

        self.load_signals()

    def create_noise_signal(self, t):

        num_sines = 10
        period = np.array([np.random.uniform(1, 20) for _ in range(num_sines)])
        period_steps = period / param.dt
        amplitude = period / 20 # amplitude between 0 and 1
        phase = np.array([np.random.uniform(0, 2*np.pi) for _ in range(num_sines)])  # random phase between 0 and 2*pi

        noisy_signal = 0

        # Add sine waves with different frequencies and amplitudes
        for i in range(num_sines):

            
            sine_wave = amplitude[i] * np.sin(((2*np.pi) / period_steps[i]) * t + phase[i])
            noisy_signal += sine_wave

        noisy_signal /= np.max(noisy_signal)

        return noisy_signal
    
    def load_signals(self):

        self.Fx = self.create_noise_signal(self.t)
        self.Fy = self.create_noise_signal(self.t)
        self.Fz = self.create_noise_signal(self.t)
        self.Tx = self.create_noise_signal(self.t)
        self.Ty = self.create_noise_signal(self.t)
        self.Tz = self.create_noise_signal(self.t)


    def calculate_force(self, t):

        t = int(t)

        max_F = 0.01 # N

        Fx = self.Fx[t] * max_F
        Fy = self.Fy[t] * max_F
        Fz = self.Fz[t] * 0

        Fd = np.array([Fx, Fy, Fz])

        return Fd

    def calculate_torque(self, t):

        t = int(t)

        max_T = 0.01 # N

        Tx = 0
        Ty = 0
        Tz = self.Tz[t] * max_T

        Td = np.array([Tx, Ty, Tz])

        return Td
    

class SpaceEnvironment():
    def __init__():
        pass
    def calculate_force(self):
        pass
    def calculate_torque(self):
        pass

class Disturbances():
    def __init__(self, dt):

        self.noise = RandomSineNoise(dt)

    def calculate_disturbance_force(self, p, q, t):

        Fd = self.noise.calculate_force(t)
        # Constant Force (uncomment next line)
        #Fd = np.array([0.1,0,0])

        return Fd
    
    def calculate_disturbance_torque(self,p, q, t):

        Td = self.noise.calculate_torque(t)

        # Constant Torque (uncomment next line)
        #Td = np.array([0,0,0])

        return Td

if __name__ == "__main__":

    ts = np.linspace(0, int(param.simulation_time / param.dt)-1, int(param.simulation_time / param.dt))

    noise = RandomSineNoise(param.dt)
    disturbances_F = []

    for t in ts:

        Fd = noise.calculate_force(t)
        disturbances_F.append(Fd)

    disturbances_F = np.array(disturbances_F)

    plt.plot(ts*param.dt, disturbances_F[:, 0])
    plt.plot(ts*param.dt, disturbances_F[:, 1])
    plt.show()



