import time
import numpy as np
import matplotlib.pyplot as plt
from pyaocs.utils import Quaternion_to_Euler321
import os

from pyaocs.simulation.non_linear_equations import NonLinearPropagator
from pyaocs import parameters as param

class SatelliteEnv():
    
    def __init__(self, render = True, bullet = False, real_time = True, use_disturbances = False, trajectory = None):

        self.render_GUI = render
        self.pybullet = bullet # Wether to use or not Pybullet
        self.realTime = real_time # If True simulation runs in real time. Change to False if training reinforcement learning.
        self.use_disturbances = use_disturbances
        self.trajectory = trajectory

        if self.pybullet or self.render_GUI:
            import pybullet as pb
            import pybullet_data
            self.pb = pb
            self.pybullet_data = pybullet_data

        if self.render_GUI:

            username = os.environ.get('USERNAME')
            if username == "Nesto":
                physicsClient = self.pb.connect(self.pb.GUI, options= ' --opengl2')# p.GUI or p.DIRECT for non-graphical version

            else:
                physicsClient = self.pb.connect(self.pb.GUI)# p.GUI or p.DIRECT for non-graphical version

           # self.pb.configureDebugVisualizer(self.pb.COV_ENABLE_GUI,0)

            if self.trajectory is not None:
                for point in self.trajectory["target_positions"].T:
                    self.pb.addUserDebugLine(point, point+0.01, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)

                # to run faster later when we plot one point at a time, as we only care about position.
                self.trajectory = self.trajectory["target_positions"].T

        elif not self.render_GUI and self.pybullet:
            physicsClient = self.pb.connect(self.pb.DIRECT)# p.GUI or p.DIRECT for non-graphical version

        if self.pybullet or self.render_GUI:
            self.pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
            self.pb.setGravity(0,0,0)
            planeId = self.pb.loadURDF("plane.urdf") # creates the square plane in simulation.
            self.boxId = self.pb.loadURDF(os.path.join("pyaocs", "urdf_models", "cylinder.urdf")) # load the object and set the pos and orientation

        self.initialize()

        # Change dynamics of system and add correct moments of inertia
        I = param.I
        self.mass = param.m
        print(f"Mass of satellite: {self.mass} kg")

        if self.pybullet:
            self.pb.changeDynamics(self.boxId, -1, mass = self.mass, linearDamping = 0, angularDamping = 0, localInertiaDiagonal = I)
            #self.mass, _, self.CoG, *_ = self.pb.getDynamicsInfo(self.boxId, -1)

        self.F = param.F # force applied by each thruster in Newtons
        self.thrust_vector_range = param.thrust_vector_range # Range that the thruster can thrust vector in degrees.
        self.fps = param.fps # Frames per second of the simulation. Mantain to 240 to mantain accuracy.
        self.simulation_time = param.simulation_time # max duration of the simulation before reset
        self._end_step = (self.fps * self.simulation_time) -1 # last timestep of the simulation.
        self.dt = 1 / self.fps
        if self.pybullet:
            self.pb.setTimeStep(self.dt) # If timestep is set to different than 240Hz, it affects the solver and other changes need to be made (check pybullet Documentation)

        self.thruster_positions = np.array(param.thruster_positions)

        if not self.pybullet:
            self.nl = NonLinearPropagator(dt = self.dt, y0=self.y0, use_disturbances=self.use_disturbances)

        # For reinforcement Leanring
        #self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,))
    
    def initialize(self):
        """Function to be runned everytime the class is initialized and the environment is reset. All the variables are reinitialized and the simulation starts from 0.
        """
        self.control = False # Wether to send control command or not (True or False)
        self._done = False  #If simulation has ended --> done = True
        self._current_step = 0 #Current step equals 0 at the beginning
        self.reward = 0 # For RL purposes, starts with 0.
        self.firing_time = 0 # Time the thrusters are on
        self.start_position = param.start_position # define the cube start position
        self.start_velocity = param.start_velocity
        self.start_orientation = param.start_orientation # define start orientation
        self.start_angular_velocity = param.start_angular_velocity # in terms of euler angles

        self.y0 = np.concatenate((self.start_position, self.start_velocity, self.start_orientation, self.start_angular_velocity))
        
        self.target_position = param.target_position # Target position where we want the satellite to move.
        self.target_orientation = param.target_orientation  # [φ,θ,ψ]
        self.previous_position = self.start_position # Previous position equals starting position at the beginning.
        self.current_position = self.start_position # Current position equals starting position at the beginning.
        self.current_orientation = self.start_orientation # Current orientation equals starting orientation at the beginning.
        self.current_velocity = self.start_velocity  # Current velocity equals starting velocity at the beginning.
        
        if self.pybullet:
            self.pb.resetBasePositionAndOrientation(self.boxId, self.current_position, self.current_orientation) # Set the position of the satellite.

        self.actual_positions = []
        self.target_positions = []
        self.actual_orientations = []
        self.target_orientations = []

        # Store 6 system parameters.
        self.F1s = []
        self.F2s = []
        self.γ1s = []
        self.γ2s = []
        self.δ1s = []
        self.δ2s = []

        self.parameters = [0,0,0,0,0,0]

        if self.render_GUI:
            # Draw the line that should be followed by satellite
            #self.pb.addUserDebugLine(self.current_position, self.target_position, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
            pass
    
    def reset_fps(self, fps):
        """Change the timestep of the simulation.

        :param fps: New timestep
        :type fps: float
        """
        self.fps = fps
        self.dt = 1 / self.fps
        self._end_step = (self.fps * self.simulation_time) -1 # last timestep of the simulation.
        if self.pybullet:
            self.pb.setTimeStep(self.dt)
        if not self.pybullet:
            self.nl.dt = self.dt
        
    def reset(self, seed = None):
        """Reset the environment and return the initial observation

        :param seed: Some variable that we are not using, but is required for RL, defaults to None
        :type seed: _type_, optional
        :return: (observation, info) 
        :rtype: _type_
        """
        # Reset the environment and return the initial observation
        #p.resetSimulation()
        self.initialize()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        """Get current state of the simulation, we choose to return the current position, current orientation and target position, but we can return whatever we want.

        :return: obs
        :rtype: array with the desired info.
        """

        # Data type of Spaces.Box is float32.
        """
        obs = np.concatenate((self.current_position,            # obs[0:3]
                              self.current_orientation,         # obs[3:7]
                              self.target_position,             # obs[7:10]
                              self.target_orientation,          # obs[10:13]
                              self.current_velocity,            # obs[13:16]
                              self.parameters)                  # obs[16:22]
                              ).astype(np.float32)
        """
        obs = {"current_position": np.array(self.current_position),
               "current_orientation": np.array(self.current_orientation),
               "target_position": np.array(self.target_position),
               "target_orientation": np.array(self.target_orientation),
               "current_velocity": np.array(self.current_velocity),
               "parameters": np.array(self.parameters)}

        return obs
 
    def _get_info(self):
        """Function to return information about the system.

        :return: You can choose what to return.
        :rtype: dict
        """
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))
        
        return {'Current Position (m)': self.current_position, "Target Position (m)": self.target_position, "Distance from target": distance_from_target, "Rewards": self.reward}
    
    def step(self, action):
        """Function to be run for every timestep of the simulation. Every time the function is run, the simulation runs one timestep of length dt, defined in __init__. Most important function in the class. 

        :param action: Variable that the control algorithm gives to the simulation to set the actions performed by the system. It's composed of 6 variables normalized between -1 and 1: 
            - F1: a value of 1 means thruster 1 should be activated, -1 means it should be off. In between depends on the scheme decided.
            - F2: a value of 1 means thruster 2 should be activated, -1 means it should be off. In between depends on the scheme decided.
            - γ1: angle y in degrees of thruster 1, should be multiplied by the range of motion of the compliant mechanism.
            - γ2: angle y in degrees of thruster 2
            - δ1: angle z in degrees of thruster 1
            - δ2: angle z in degrees of thruster 2
        :type action: _type_
        :return: _description_
        :rtype: _type_
        """

        F1, F2 = action[:2]

        γ1, γ2, δ1, δ2 = action[2:] * np.pi / 180

        # If the output signal from the PID is very small, smaller than 0.01 (for example), the thruster is not activated. This reduces fuel usage, but decreases accuracy.
        if F1 > 0.01:
            F1 = self.F
            #F = round(F1, 5)
            self.apply_force(force = F1, γ = γ1, δ = δ1, thruster_number = 1)
        else:
            self.F1 = 0

        if F2 > 0.01:
            F2 = self.F
            #F = round(F2, 5)
            self.apply_force(force = F2, γ = γ2, δ = δ2, thruster_number = 2)
        else:
            F2 = 0

        self.parameters = np.array([F1, F2, γ1, γ2, δ1, δ2])

        if self.pybullet:
            self.pb.stepSimulation()
            self.current_position, self.current_orientation = self.pb.getBasePositionAndOrientation(self.boxId)
            self.current_velocity = self.pb.getBaseVelocity(self.boxId)[0]

        else:
            p, v, q, w = self.nl.stepSimulation(self.parameters)
            self.current_position = p.copy()
            self.current_orientation = q.copy()
            self.current_velocity = v.copy()

        if self.render_GUI and self.pybullet == False:
            self.pb.resetBasePositionAndOrientation(self.boxId, p, q) # Set the position of the satellite.
            self.pb.resetBaseVelocity(self.boxId, v)

        if self.realTime:
            time.sleep(1./self.fps)

        # Save the data from the simulation. Positions, target positions, orientations
        self.parameters[2:] = self.parameters[2:] * 180 / np.pi
        self.data_recorder(self.parameters)
        
        reward = self.reward_calculation()
        self.reward += reward

        self.previous_position = self.current_position
            
        if self._current_step == self._end_step:
            self._done = True
            observation = self._get_obs()
            #print(observation)
            info = self._get_info()
            #print(info)
        
        else:   
            observation = self._get_obs()
            info = self._get_info()

        self._current_step += 1
        
        return observation, reward, self._done, self.control, info
    
    def get_force_vector(self, force, γ, δ, thruster_number = 1):
        """Given the thruster angles and force, find the force vector.

        :param force: Force applied by the thruster
        :type force: float
        :param γ: the angle in radians that the thruster is vectored about the y-axis.
        :type γ: float
        :param δ: the angle in radians that the thruster is vectored about the z-axis. 
        :type δ: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        opposite_y = force * np.sin(δ)
        opposite_z = force * np.sin(γ)
        opposite_total = np.sqrt(opposite_y**2 + opposite_z**2)
        resultant_x = np.sqrt(force**2 - opposite_total**2)

        force_magnitude_check = np.sqrt(resultant_x**2 + opposite_y**2 + opposite_z**2)

        assert round(force_magnitude_check, 5) == force, f"Component forces {force_magnitude_check} do not equate to initial force {force}."

        # In the x_direction the resultant_x should have different sign. Unless you don't want one of the thrusters to activate.
        # In the y_direction the opposite_y should have opposite signs for rotation and equal sign for translation.
        # In the z_direction the opposite_z should have opposite signs for rotation and equal sign for translation.

        if thruster_number == 1:
            self.force_vector = np.array([resultant_x, opposite_y, -opposite_z])

        elif thruster_number == 2:
            # Thruster is placed on opposite direction to thruster 1, so the sign of resultant_x is reversed
            self.force_vector = np.array([-resultant_x, -opposite_y, -opposite_z])

        self.thruster_position = self.thruster_positions[thruster_number - 1] # Position of thruster with respect to satellite origin.
    
    def apply_force(self, force: float, γ: float, δ: float, thruster_number: int = 0) -> None:
        """Function to apply force by the thruster in a given timestep

        :param force: Force applied by the thruster
        :type force: float
        :param γ: the angle in radians that the thruster is vectored about the y-axis.
        :type γ: float
        :param δ: the angle in radians that the thruster is vectored about the z-axis. 
        :type δ: float
        :param thruster_number: The number of the thruster you want to activate, defaults to 1
        :type thruster_number: int, optional
        """

        self.get_force_vector(force, γ, δ, thruster_number)

        if self.pybullet:
            self.pb.applyExternalForce(self.boxId, -1, self.force_vector, self.thruster_position, self.pb.LINK_FRAME) # in Newton # WORLD_FRAME p.LINK_FRAME
            #p.applyExternalForce(self.boxId, -1, (-resultant_x, -opposite_y, -opposite_z), [-0.5,0,0], p.LINK_FRAME)

        if self.render_GUI:
            joint_index = thruster_number - 1 # Index of the joint you want to control
            target_angle = δ # Target angle in radians (for example, 90 degrees)
            self.pb.resetJointState(bodyUniqueId=self.boxId, jointIndex=joint_index, targetValue=target_angle)


        self.draw_thrust_trajectory()

        self.firing_time += self.dt

    def reward_calculation(self):
        """Defines how the rewards for RL are calculated. Many methods can be used.

        :return: reward
        :rtype: int
        """

        distance_from_target_prev = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.previous_position))**2))
        distance_from_target = np.sqrt(np.sum((np.array(self.target_position) - np.array(self.current_position))**2))

        #reward =  50000*(distance_from_target_prev - distance_from_target) / distance_from_target
        
        if distance_from_target <= distance_from_target_prev:
            reward = 1 / distance_from_target
        else:
            reward = 0
        

        return reward

    def draw_thrust_trajectory(self):
        """Draw the red lines representing the thrust.
        """
        if self.render_GUI:
            line_lenght = 0.2
            # Remember self.force_vector is defined in get_force_vector
            self.pb.addUserDebugLine(self.thruster_position, self.thruster_position - line_lenght*self.force_vector, lineColorRGB=[1.00,0.25,0.10], lineWidth=5.0, lifeTime=self.dt, parentObjectUniqueId = self.boxId)

        # show all the trajectory points at their correpsonding time
        if self.render_GUI and self.trajectory is not None and self._current_step % param.step_rate == 0:

            point = self.trajectory[int(self._current_step / param.step_rate)]
            self.pb.addUserDebugLine(point, point+0.02, lineColorRGB=[0, 1, 0], lineWidth=4.0, lifeTime=0)

    
    def data_recorder(self, inputs):
        """Saves important data every timestep for visualization after the simulation finishes, like the x, y, z position coordinates.
        """
        self.actual_positions.append(self.current_position)
        self.target_positions.append(self.target_position)

        #rotation = Rotation.from_quat(self.current_orientation)
        #euler_angles = rotation.as_euler('zyx', degrees=True)
        euler_angles = Quaternion_to_Euler321(self.current_orientation)

        self.actual_orientations.append(euler_angles)
        self.target_orientations.append(self.target_orientation)

        F1, F2, γ1, γ2, δ1, δ2 = inputs

        self.F1s.append(F1)
        self.F2s.append(F2)
        self.γ1s.append(γ1)
        self.γ2s.append(γ2)
        self.δ1s.append(δ1)
        self.δ2s.append(δ2)

    def render(self, mode = 'human'):
        """This is required for the RL library. Not used if not.

        :param mode: _description_, defaults to 'human'
        :type mode: str, optional
        """
        self.render_GUI = True
        return
        
    def plot_training(self):
        """If RL used, this function plots important things that we would like to see, like episode rewards.
        """
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_balances)
        plt.title("Balances (£)")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_trades)
        plt.title("Number of trades")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_success)
        plt.title("Successful trades (%)")
        
        plt.show()
        
    
    def plot_results(self, name = "performance_report"):
        """Plots the data recorded.

        :param name: _description_, defaults to "performance_report"
        :type name: str, optional
        """

        self.t = np.linspace(0, (self._current_step + 1) / self.fps, self._current_step)

        self.plot_pose()
        self.plot_parameters()
        if self.pybullet == False:
            self.plot_disturbances()

        print(f"Total firing time: {round(self.firing_time, 3)} s")

        return
    
    def plot_pose(self):

        self.actual_positions = np.array(self.actual_positions)
        self.target_positions = np.array(self.target_positions)
        self.actual_orientations = np.array(self.actual_orientations)
        self.target_orientations = np.array(self.target_orientations)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.actual_positions[:, 0],  label='Actual')
        #plt.plot(self.t, self.target_positions[:, 0], label='Target')
        plt.title("X positions")
        plt.xlabel("t (s)")
        plt.ylabel("x (m)")

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.actual_positions[:, 1],  label='Actual')
       # plt.plot(self.t, self.target_positions[:, 1], label='Target')
        plt.title("Y positions")
        plt.xlabel("t (s)")
        plt.ylabel("y (m)")

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.actual_positions[:, 2],  label='Actual')
        #plt.plot(self.t, self.target_positions[:, 2], label='Target')
        plt.title("Z positions")
        plt.xlabel("t (s)")
        plt.ylabel("z (m)")

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.actual_orientations[:,2],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,2], label='Target')
        plt.title("φ positions")
        plt.xlabel("t (s)")
        plt.ylabel("φ (°)")

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.actual_orientations[:,1],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,1], label='Target')
        plt.title("θ positions")
        plt.xlabel("t (s)")
        plt.ylabel("θ (°)")

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.actual_orientations[:,0],  label='Actual')
        #plt.plot(self.t, self.target_orientations[:,0], label='Target')
        plt.title("ψ positions")
        plt.xlabel("t (s)")
        plt.ylabel("ψ (°)")

        plt.tight_layout()
        plt.show()
    
    def plot_parameters(self):
        """Plot the 6 system parameters F1, F2, γ1, γ2, δ1, δ2 over time"""
    
        self.F1s = np.array(self.F1s)
        self.F2s = np.array(self.F2s)
        self.γ1s = np.array(self.γ1s)
        self.γ2s = np.array(self.γ2s)
        self.δ1s = np.array(self.δ1s)
        self.δ2s = np.array(self.δ2s)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.F1s,  label='F1')
        plt.title("F1 thrust")
        plt.xlabel("t (s)")
        plt.ylabel("F (N)")
        plt.grid()

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.F2s, label='F2')
        plt.title("F2 thrust")
        plt.xlabel("t (s)")
        plt.ylabel("F (N)")
        plt.grid()

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.γ1s,  label='γ1')
        plt.title("γ1 angle")
        plt.xlabel("t (s)")
        plt.ylabel("γ1 (°)")
        plt.grid()

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.γ2s, label='γ2')
        plt.title("γ2 angle")
        plt.xlabel("t (s)")
        plt.ylabel("γ2 (°)")
        plt.grid()

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.δ1s,  label='δ1')
        plt.title("δ1 angle")
        plt.xlabel("t (s)")
        plt.ylabel("δ1 (°)")
        plt.grid()

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.δ2s, label='δ2')
        plt.title("δ2 angle")
        plt.xlabel("t (s)")
        plt.ylabel("δ2 (°)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_disturbances(self):

        self.nl.disturbances_F = np.array(self.nl.disturbances_F)
        self.nl.disturbances_T = np.array(self.nl.disturbances_T)

        plt.subplot(3, 2, 1)
        plt.plot(self.t, self.nl.disturbances_F[:, 0])
        plt.title("x force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fx (N)")

        plt.subplot(3, 2, 2)
        plt.plot(self.t, self.nl.disturbances_F[:, 1])
        plt.title("y force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fy (N)")

        plt.subplot(3, 2, 3)
        plt.plot(self.t, self.nl.disturbances_F[:, 2])
        plt.title("z force disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Fz (N)")

        plt.subplot(3, 2, 4)
        plt.plot(self.t, self.nl.disturbances_T[:,0])
        plt.title("φ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tφ (Nm)")

        plt.subplot(3, 2, 5)
        plt.plot(self.t, self.nl.disturbances_T[:,1])
        plt.title("θ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tθ (Nm)")

        plt.subplot(3, 2, 6)
        plt.plot(self.t, self.nl.disturbances_T[:,2])
        plt.title("ψ torque disturbance")
        plt.xlabel("t (s)")
        plt.ylabel("Tψ (Nm)")

        plt.tight_layout()
        plt.show()

    def close(self):
        """Run to close the simulation window
        """
        if self.pybullet or self.render_GUI:
            self.pb.disconnect()