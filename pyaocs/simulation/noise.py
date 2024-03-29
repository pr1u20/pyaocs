import numpy as np
import matplotlib.pyplot as plt
from pyaocs import parameters as param
from pyaocs.utils import Quaternion_to_Euler321, euler_to_quaternion, normalize_quaternion, ExponentialMovingAverageFilter

class Noise():
    def __init__(self, sample = "100hz"):

        duration = param.simulation_time   # Duration in seconds

        self.sample = sample

        if self.sample == "24hz":
            sample_rate = 24
        else:
            sample_rate = param.fps  # Sampling rate in Hz

        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        self.step = 0

        if self.sample == "24hz":
            self.position_noise = self.generate_signal(1, 0.00005)
            self.orientation_noise = self.generate_signal(1, 0.00005)
        else:
            self.position_noise = self.generate_signal(1, 0.00001)
            self.orientation_noise = self.generate_signal(1, 0.00001)


    def add_noise(self, obs):
        """
        Add noise to the IMU sensor data.
        """
        p = obs["current_position"]
        q = obs["current_orientation"]

        p = p + self.position_noise[self.step]

        q = q + self.orientation_noise[self.step]
        q = normalize_quaternion(q)

        obs["current_position"] = p
        obs["current_orientation"] = q

        self.step += 1

        return obs
    
    def generate_signal(self, frequency, amplitude):
        """
        Generate a sine wave signal with the given parameters.
        """
        # Generate a sine wave
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * self.t)

        if self.sample:
            coefficient = (amplitude * 10)
        else:
            coefficient = (amplitude * 50)

        # Add Gaussian noise
        noise = coefficient*np.random.normal(0, 0.2, sine_wave.shape)   # Noise with mean=0 and std dev=0.2
        noisy_signal = sine_wave + noise

        return noisy_signal




if __name__ == "__main__":

    noise = Noise("24hz")
    ema = ExponentialMovingAverageFilter(alpha = 0.1)

    obs = {"current_position": np.array([0, 0, 0]),
           "current_orientation": euler_to_quaternion(np.array([0, 0, 0]))}
    
    positions = []
    orientations = []
    filtered_positions = []

    num_steps = 200

    for i in range(num_steps):
        obs = noise.add_noise(obs)
        filtered = ema.filter(obs["current_position"])
        filtered_positions.append(filtered)
        positions.append(obs["current_position"])
        orientations.append(Quaternion_to_Euler321(obs["current_orientation"]))

    # Make to subfigures for position and orientation
    positions = np.array(positions).T
    orientations = np.array(orientations).T
    filtered_positions = np.array(filtered_positions).T

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(noise.t[:num_steps], positions[0], label="X")
    axs[0].plot(noise.t[:num_steps], filtered_positions[0], label="Filtered X")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(noise.t[:num_steps], orientations[0], label="Yaw")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Orientation")
    axs[1].legend()

    plt.show()    
