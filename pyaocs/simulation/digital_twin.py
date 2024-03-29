import time
import numpy as np
import traceback
from pyaocs import parameters as param
from pyaocs.utils import wait_for_actuator
from pyaocs.simulation.environment import SatelliteEnv
from pyaocs.simulation.actuator import ActuatorSimulation
from pyaocs.simulation.noise import Noise


def run(strategy, render = True, real_time = True, bullet = False, use_disturbances = False, plot = True, trajectory = None, noise = False, wait_actuator = True):

    env = SatelliteEnv(render = render, bullet = bullet, real_time=real_time, use_disturbances=use_disturbances, trajectory=trajectory)
    obs, info = env.reset()

    act = ActuatorSimulation()

    if noise:
        QTM_noise = Noise()

    step_rate = param.step_rate # Steps after which control function is run.

    done = False
    try:
        while not done:
            # Choose an action
            if env._current_step % step_rate == 0:

                t1 = time.time()

                # AOCS control block
                action = strategy.compute(obs) # Gives target values

                time.sleep(0.000001)
                t2 = time.time()

                fps = 1 / (t2 - t1)
                print(f"FPS: {fps:.2f}", end='\r', flush=True)

            # Actuator control block
            state, finished = act.compute(action) # Gives variables values after step change.

            # If actuator has not reached target, do not fire thruster
            if wait_actuator:
                action, state = wait_for_actuator(state, action)

            # Perform the chosen action in the environment
            obs, reward, done, _, info = env.step(state)

            # Add noise to the observation
            if noise:
                obs = QTM_noise.add_noise(obs)

        print(f"FPS: {fps:.2f}")
        env.close()

    except KeyboardInterrupt:
        traceback.print_exc()
        print("An error occurred. Exiting simulation.")

    if plot:
        env.plot_results()
        strategy.plot()

    return env, obs, strategy