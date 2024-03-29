import numpy as np

from pyaocs import parameters as param


class ActuatorSimulation():
    """Class used to change the 6 parameters of the satellite F1, F2, γ1, γ2, δ1, δ2. Details of the variables and order are specified in the Guidelines 55 Requirements document.
    Would not be neccessary if we assumed instaneous changes, but there are delays, limitations in actuation speed, etc.
    For now we will assume linear changes, in the future we should have more complex estimations.
    This is external from the Satellite Env class as this will be an input from a different work package, where they could specify the exact relationship between torque and angle change, the actual change in thrust"""

    def __init__(self, dt = param.dt):

        self.initialize()

        self.actuation_speed = param.actuation_speed # Rate of change of γ1, γ2, δ1, δ2
        self.thrust_speed = param.thrust_speed # Rate of change of F1, F2

        self.dt = dt

    def initialize(self):

        """In case Reinforcement Learning is used, the initialize function needs to be run after every episode. To reset variables.
        Any variable that changes through time, should be defined here.
        """

        # The six system parameters that can be changed to control satellite.
        self.F1 = 0
        self.F2 = 0
        self.γ1 = 0
        self.γ2 = 0
        self.δ1 = 0
        self.δ2 = 0

    def compute(self, targets, wait_actuator = False):

        """Computes the next position of the satellite parameters given the target value

        :param targets: 6 target values for the state variables of the satellite.
        :param type: np.ndarray
        :return: (state, finished).
            -   state (np.ndarray): the updated six parameters of the satellite.
            -   finished (np.ndarray): whether each of the six variables has reached there target values.
                Consists of a numpy array with 6 bool variables. 
        :rtype: float
        """

        target_F1 = targets[0]
        target_F2 = targets[1]
        target_γ1 = targets[2]
        target_γ2 = targets[3]
        target_δ1 = targets[4]
        target_δ2 = targets[5]

        # Calculate change for each parameter
        self.F1, F1_finshed = self.actuation_change(self.F1, target_F1, self.thrust_speed)
        self.F2, F2_finshed = self.actuation_change(self.F2, target_F2, self.thrust_speed)
        self.γ1, γ1_finshed = self.actuation_change(self.γ1, target_γ1, self.actuation_speed)
        self.γ2, γ2_finshed = self.actuation_change(self.γ2, target_γ2, self.actuation_speed)
        self.δ1, δ1_finshed = self.actuation_change(self.δ1, target_δ1, self.actuation_speed)
        self.δ2, δ2_finshed = self.actuation_change(self.δ2, target_δ2, self.actuation_speed)

        state = np.array([self.F1, self.F2, self.γ1, self.γ2, self.δ1, self.δ2])
        finished = np.array([F1_finshed, F2_finshed, γ1_finshed, γ2_finshed, δ1_finshed, δ2_finshed])

        return state, finished

    def actuation_change(self, value, target_value, rate_of_change):
        
        """Common function used to vary all the parameters. In the future thrust will not changes as the thruster angles, so there will be different functions for each.

        :param value: current values of the system
        :type value: float
        :param target_value: target values for the system
        :type target_value: float
        :param rate_of_change: rate of change of the variable
        :type rate_of_change: float
        :return: (value, finished)
            -   state (np.ndarray): the updated value.
            -   finished (np.ndarray): whether the value is equal to target values. 
        :rtype: float
        """

        #Whether value == target_value, so objective target has been reached
        finished = False

        if value == target_value:
            finished = True
            pass

        elif value < target_value:

            value = value + rate_of_change*self.dt

            if value > target_value:
                value = target_value
                finished = True

        elif value > target_value:

            value = value - rate_of_change*self.dt

            if value < target_value:
                value = target_value
                finished = True

        return value, finished