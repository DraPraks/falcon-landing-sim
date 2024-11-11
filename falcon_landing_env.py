import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from typing import Optional

class FalconLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(FalconLandingEnv, self).__init__()
        self.render_mode = render_mode
        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([0, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        self._connect_pybullet(mode='GUI' if self.render_mode == 'human' else 'DIRECT')
        self.reset()

        self.time_step = 1./240.  # PyBullet's default time step
        p.setTimeStep(self.time_step)

    def _connect_pybullet(self, mode='DIRECT'):
        if not hasattr(self, 'physicsClient'):
            if mode == 'GUI':
                self.physicsClient = p.connect(p.GUI)
            else:
                self.physicsClient = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Handle the seed
        super().reset(seed=seed)
        p.resetSimulation()
        planeId = p.loadURDF("plane.urdf")
        self.rocketId = p.loadURDF("rocket.urdf", [0, 0, 100])
        self.fuel = 1000  # Arbitrary fuel units
        self.done = False

        # Get the initial observation
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            reward = 0
            info = {}
            terminated = True
            truncated = False  # Use True if you have a time limit and it's reached
            return observation, reward, terminated, truncated, info

        # Parse action
        throttle = action[0]  # Main engine throttle
        gimbal_pitch = action[1]
        gimbal_yaw = action[2]
        fin_pitch = action[3]
        fin_yaw = action[4]

        # Apply physics
        self._apply_engine_force(throttle, gimbal_pitch, gimbal_yaw)
        self._apply_control_surfaces(fin_pitch, fin_yaw)
        self._apply_wind()

        # Step simulation
        p.stepSimulation()

        observation = self._get_observation()
        reward = self._calculate_reward(observation, action)
        self.done = self._check_done(observation)
        terminated = self.done
        truncated = False  # Update this based on your environment logic
        info = {}
        return observation, reward, terminated, truncated, info

    def _apply_engine_force(self, throttle, gimbal_pitch, gimbal_yaw):
        if self.fuel <= 0:
            throttle = 0  # No fuel means no thrust

        # Calculate thrust
        max_thrust = 760700  # Newtons, for Falcon 9's Merlin engine
        thrust = throttle * max_thrust

        # Consume fuel
        fuel_consumption_rate = 250  # Arbitrary units per second
        self.fuel -= throttle * fuel_consumption_rate * self.time_step
        self.fuel = max(self.fuel, 0)  # Prevent negative fuel

        # Apply thrust force
        # Get orientation quaternion
        orientation = p.getBasePositionAndOrientation(self.rocketId)[1]
        # Calculate thrust direction with gimbal adjustments
        thrust_direction = self._calculate_thrust_direction(orientation, gimbal_pitch, gimbal_yaw)
        force = [thrust * d for d in thrust_direction]

        # Apply force to the rocket's center of mass
        p.applyExternalForce(self.rocketId, -1, force, [0, 0, 0], p.LINK_FRAME)

    def _calculate_thrust_direction(self, orientation, gimbal_pitch, gimbal_yaw):
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        # Adjust for gimbal angles
        # Compute new thrust direction based on gimbal_pitch and gimbal_yaw
        # For simplicity, let's assume small angles
        thrust_direction = [0, 0, 1]  # Upwards in local frame
        # Apply gimbal adjustments
        thrust_direction[0] += gimbal_pitch
        thrust_direction[1] += gimbal_yaw
        # Normalize vector
        norm = np.linalg.norm(thrust_direction)
        if norm == 0:
            return [0, 0, 0]
        thrust_direction = [d / norm for d in thrust_direction]
        return thrust_direction

    def _apply_control_surfaces(self, fin_pitch, fin_yaw):
        # Calculate aerodynamic forces
        velocity = p.getBaseVelocity(self.rocketId)[0]
        speed = np.linalg.norm(velocity)
        air_density = 1.225  # kg/m^3 at sea level
        drag_coefficient = 1.5  # Approximate value
        fin_area = 1.0  # m^2, adjust as needed

        # Calculate drag force
        drag_force_magnitude = 0.5 * air_density * speed**2 * drag_coefficient * fin_area

        # Direction opposite to velocity
        if speed != 0:
            drag_force_direction = [-v / speed for v in velocity]
        else:
            drag_force_direction = [0, 0, 0]
        # Adjust for fin angles
        drag_force_direction[0] += fin_pitch
        drag_force_direction[1] += fin_yaw
        # Normalize
        norm = np.linalg.norm(drag_force_direction)
        if norm == 0:
            drag_force = [0, 0, 0]
        else:
            drag_force = [drag_force_magnitude * d / norm for d in drag_force_direction]

        # Apply drag force
        p.applyExternalForce(self.rocketId, -1, drag_force, [0, 0, 0], p.WORLD_FRAME)

    def _apply_wind(self):
        # Define wind profile
        wind_speed = [5, 0, 0]  # Wind blowing along the x-axis at 5 m/s
        # Calculate relative wind speed
        rocket_velocity = p.getBaseVelocity(self.rocketId)[0]
        relative_wind_speed = [wind_speed[i] - rocket_velocity[i] for i in range(3)]
        speed = np.linalg.norm(relative_wind_speed)
        if speed == 0:
            wind_force = [0, 0, 0]
        else:
            # Calculate wind force
            air_density = 1.225  # kg/m^3
            drag_coefficient = 1.0  # Simplified value
            rocket_cross_sectional_area = 10.0  # m^2, adjust as needed
            wind_force_magnitude = 0.5 * air_density * speed**2 * drag_coefficient * rocket_cross_sectional_area
            # Direction of wind force
            wind_force_direction = [v / speed for v in relative_wind_speed]
            wind_force = [wind_force_magnitude * d for d in wind_force_direction]

        # Apply wind force
        p.applyExternalForce(self.rocketId, -1, wind_force, [0, 0, 0], p.WORLD_FRAME)

    def _calculate_reward(self, obs, action):
        # Extract relevant state variables
        position = p.getBasePositionAndOrientation(self.rocketId)[0]
        velocity = p.getBaseVelocity(self.rocketId)[0]
        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.rocketId)[1])

        # Parameters
        target_position = [0, 0, 0]  # Landing pad at origin
        landing_zone_radius = 5.0  # Acceptable landing zone radius in meters

        # Compute distance to target
        distance = np.linalg.norm([position[0] - target_position[0], position[1] - target_position[1]])

        # Reward components
        distance_reward = -distance
        vertical_velocity_reward = -abs(velocity[2])
        horizontal_velocity_reward = -np.linalg.norm([velocity[0], velocity[1]])
        orientation_reward = -np.linalg.norm([orientation[0], orientation[1]])  # Penalize tilting
        fuel_penalty = -action[0] * 0.1  # Penalize fuel usage

        # Total reward
        reward = distance_reward + vertical_velocity_reward + horizontal_velocity_reward + orientation_reward + fuel_penalty

        # Bonus for successful landing
        if self._check_landed(position, velocity):
            reward += 1000  # Large positive reward

        return reward

    def _check_landed(self, position, velocity):
        landing_pad_z = 0  # Ground level
        landed = position[2] <= landing_pad_z + 1 and abs(velocity[2]) < 1
        return landed

    def _check_done(self, obs):
        position = p.getBasePositionAndOrientation(self.rocketId)[0]
        if position[2] < 0:
            # Rocket has crashed below ground level
            return True
        elif self._check_landed(position, p.getBaseVelocity(self.rocketId)[0]):
            # Rocket has landed successfully
            return True
        else:
            return False

    def _get_observation(self):
        # Get position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.rocketId)
        position = np.array(position, dtype=np.float32)  # Ensure dtype is float32

        velocity, angular_velocity = p.getBaseVelocity(self.rocketId)
        velocity = np.array(velocity, dtype=np.float32)
        angular_velocity = np.array(angular_velocity, dtype=np.float32)

        orientation_euler = p.getEulerFromQuaternion(orientation)
        orientation_euler = np.array(orientation_euler, dtype=np.float32)

        fuel_level = np.array([self.fuel], dtype=np.float32)

        # Construct observation
        observation = np.concatenate([
            position,          # x, y, z (3 elements)
            velocity,          # vx, vy, vz (3 elements)
            orientation_euler, # roll, pitch, yaw (3 elements)
            angular_velocity,  # angular velocities (3 elements)
            fuel_level         # fuel level (1 element)
        ])

        return observation

    def render(self):
        if self.render_mode == 'human':
            if not hasattr(self, 'physicsClient'):
                self._connect_pybullet(mode='GUI')
            # PyBullet GUI will handle rendering
        elif self.render_mode == 'rgb_array':
            # Return an image array
            width, height, rgbPixels, _, _ = p.getCameraImage(
                width=640,
                height=480,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(rgbPixels, dtype=np.uint8).reshape((height, width, 4))
            return rgb_array[:, :, :3]  # Return RGB channels
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not implemented")
