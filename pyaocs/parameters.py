start_position = [0, 0, 1] # define the cube start position
start_velocity = [0, 0, 0]
start_orientation = [0.0, 0.0, 0.0, 1.0] # define start orientation [0.0, 0.0, 0.4214104, 0.90687 ]
start_angular_velocity = [0, 0, 0] # in terms of euler angles

target_position = [2,2,1] # Target position where we want the satellite to move.
target_orientation = [0,0,0]  # [ψ, θ, φ]


thruster_positions = [[-0.15,0,0], 
                      [0.15,0,0]]

F = 0.26 # force applied by each thruster in Newtons. F/m ratio 1/20
L = abs(thruster_positions[0][0]) # m
m = 10 # kg
Ix = m * L**2  # 0.05 kg m^2
Iy = m * L**2 # 0.05 kg m^2
Iz = 0.15 # 0.1 kg m^2

I = Ix, Iy, Iz

fps = 240 # Hz. # Frames per second of the simulation. Mantain to 240 to mantain accuracy.
sample_rate = 24 # Hz. Rate at which the control algorithms reads position information.
step_rate = int(fps / sample_rate)
dt = 1 / fps
thrust_vector_range = 16 # degrees
discrete_angle = thrust_vector_range

simulation_time = 50 # s
actuation_speed = 240 # degree/s. Assuming linear, but could be some complex function dependent on current angles.
thrust_delay = 0.04 #3e-4 #30e-3 # s
thrust_speed = F / thrust_delay #N / s  How fast the thrust can be increased to desired value. We will assume it to be linear increase from 0 to desired force (1N) in the thrust delay time (30 ms)
discrete_thurster = 24 # the levels of thrust we can get 

# Orbit Fab uses rotation_error of 0.01 deg, rotation_drift of 0.1 deg/s and translation_error of 0.01 m.
translation_error = 0.05    # m
translation_drift = 0.005   # m/s
rotation_error = 0.1       # degree
rotation_drift = 0.01       # degree/s
max_v = 0.3 # m/s