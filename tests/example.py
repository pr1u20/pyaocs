import numpy as np

from pyaocs import parameters as param
from pyaocs.simulation import digital_twin

class SimpleTestControl():

    def __init__(self):
        self.firing_time = 1 #s

        self.dt = 1 / param.sample_rate

        self.total_steps = self.firing_time / self.dt

        self.step = 0

    def compute(self, obs):

        F1 = 0
        F2 = 0
        γ1 = 0
        γ2 = 0
        δ1 = 0
        δ2 = 0

        if (self.step + 1) <= self.total_steps:
            F1 = 1

        action = np.array([F1, F2, γ1, γ2, δ1, δ2])

        self.step += 1

        return action
    
    def plot(self):
        pass
    

if __name__ == "__main__":

    strategy = SimpleTestControl()

    digital_twin.run(strategy, 
                     render=True, 
                     real_time=False, 
                     use_disturbances=False,
                     noise=False)
    