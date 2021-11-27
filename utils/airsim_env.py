"""
Pos(AirSim):
    LB = (30, -40)
    LT = (70, -40)
    RT = (70, 40)
    RB = (30, 40)
    Player1 = (0, 0)
    Player2 = (0, -10)
    Player3 = (0, 10)
Pos(UE4):
    LB = (30, -50)
    LT = (70, -50)
    RT = (70, 30)
    RB = (30, 30)
    Player1 = (0, -10)
    Player2 = (0, -20)
    Player3 = (0, 0)
Landmarks:
(UE4)
    (45, -30)
    (36, -2)
    (45, 10)
(AirSim)
    (45, -20)
    (36, 8)
    (45, 20)
"""
import time
import numpy as np
import airsim
from numpy.lib.function_base import vectorize
from utils.env_core import Landmark


np.set_printoptions(precision=3, suppress=True)
clock_speed = 5
time_slice = 0.5 / clock_speed
# speed_limit = 0.5
ACTION = ['00', '-x', '+x', '-y', '+y']


class Env:
    def __init__(self):
        """
        Description:
            connect to the AirSim simulator
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.num_drones = 3
        self.num_landmarks = 3
        self.ctrl_list = []  # control buffer
        self.start_time = time.time()
        # serial number of drones
        self.drones = [('Drone' + str(i+1)) for i in range(self.num_drones)]
        self.landmarks = [Landmark() for i in range(self.num_landmarks)]

    def reset(self):
        """
        Description:
            get observation from original environment
        Outputs:
            obs per drone: (p_vel, p_pos, t_pos, o_pos)
                p_vel: private velocity
                p_pos: private position
                t_pos: relative position from the target
                o_pos: relative position from other drones
        """
        # init landmarks
        for i, landmark in enumerate(self.landmarks):
            landmark.name = "landmark %d" % i
            # pos_y = np.random.uniform(30, 70)
            # pos_x = np.random.uniform(-40, 40)
            if i == 0:
                pos_x = 45
                pos_y = -20
            elif i == 1:
                pos_x = 36
                pos_y = 8
            elif i == 2:
                pos_x = 45
                pos_y = 20
            landmark.state.p_pos = np.array([pos_x, pos_y]) #  HACK:

        # init drones
        self.client.reset()
        for drone in self.drones:
            self.client.enableApiControl(True, drone)

        # begin simulate
        self.client.simPause(False)

        # drones takeoff
        for drone in self.drones:
            self.ctrl_list.append(
                self.client.moveToZAsync(-10, 3, vehicle_name=drone))
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()

        # drones cushion
        time.sleep(3)
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()

        # drones hover
        for drone in self.drones:
            self.ctrl_list.append(self.client.hoverAsync(vehicle_name=drone))
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()
        
        # pause simulate
        self.client.simPause(True)
        
        # nagents obs, obs.shape = (nagents)(nobs), nobs differs per agent
        obs = []
        for drone in self.drones:
            p_vel = self.drone_vel(drone)
            p_pos = self.drone_pos(drone)
            t_pos = []
            for landmark in self.landmarks:
                t_pos.append(landmark.state.p_pos - self.drone_pos(drone))
            o_pos = []
            for other in self.drones:
                if other is drone: continue
                o_pos.append(self.drone_pos(drone) - self.drone_pos(other))
            obs.append(np.concatenate([p_vel] + [p_pos] + t_pos + o_pos))
        return obs

    def drone_pos(self, drone):
        """
        Description:
            get the position of specified drone
        Input:
            drone (str): drone id
        Output:
            pos (list of ints): positions(only x, y) of specified drone
        """
        pos = self.client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val])  # HACK:
        # pos corr
        if drone == "Drone1":
            pass
        elif drone == "Drone2":
            pos += np.array([0, -10])
        elif drone == "Drone3":
            pos += np.array([0, 10])
        else:
            raise Exception("Valid drone index: ", drone)
        # pos /= 100.0  # HACK:
        return pos

    def drone_vel(self, drone):
        """
        Description:
            get the velocity of specified drone
        Input:
            drone (str): drone id
        Output:
            vel (list of ints): velocity(only x, y) of specified drone
        """
        vel = self.client.getMultirotorState(vehicle_name=drone).kinematics_estimated.linear_velocity
        vel = np.array([vel.x_val, vel.y_val])  # HACK:
        return vel

    def drone_move(self, drone, quad_offset):
        """
        Description:
            move instructions
        Input:
            drone (str): drone id
            quad_offset (list of ints): velocity(only x, y)
        Output:
            client move instruction (need .join())
        """
        # HACK:
        return self.client.moveByVelocityZAsync(quad_offset[0], quad_offset[1],
                                         -10, time_slice, vehicle_name=drone)

    def step(self, actions):
        """
        Description:
            interact with environment based on velocity per drone
        Input:
            quad_offset (list of ints): velocity for all drone
        Output:
            obs (list of floats): observation for all drone
            reward (float): reward from environment
            done (bool): is done
            info (dict): information
        """
        # move with given velocity
        quad_offset = self.interpret_action(actions)
        quad_offset = [[float(i) for i in j] for j in quad_offset]
        self.client.simPause(False)
        collision = 0
        dead = False

        for i, drone in enumerate(self.drones):
            self.ctrl_list.append(self.drone_move(drone, quad_offset[i]))
        for ctrl in self.ctrl_list:
            ctrl.join()
        # for i, drone in enumerate(self.drones):
        #     self.drone_move(drone, quad_offset[i])

        start_time = time.time()
        # while time.time() - start_time < time_slice:
        #     for drone in self.drones:
        #         if self.client.simGetCollisionInfo(drone).has_collided:
        #             collision += 1
            # if collision > 10:
            #     dead = True
        # self.ctrl_list.clear()  # hack: wait time?
        # collection_count = 0

        # now_time = time.time()
        # while time.time() - now_time < time_slice:
        #     pass
        self.client.simPause(True)
        # get drone states
        n_pos = []
        n_vel = []
        obs = []
        for drone in self.drones:
            p_vel = self.drone_vel(drone)
            p_pos = self.drone_pos(drone)
            n_pos.append(p_pos)
            n_vel.append(p_vel)
            t_pos = []
            for landmark in self.landmarks:
                t_pos.append(landmark.state.p_pos - self.drone_pos(drone))
            o_pos = []
            for other in self.drones:
                if other is drone: continue
                o_pos.append(self.drone_pos(other) - self.drone_pos(drone))
            obs.append(np.concatenate([p_vel] + [p_pos] + t_pos + o_pos))
        # obs = np.array([obs])  # (1, 3, 14)
        # decide whether done
        # done = True if (time.time() - self.start_time) >= 60 else False
        done = dead

        # compute reward
        reward = self.compute_reward(n_vel, dead)

        # log info
        info = {}
        return obs, reward, done, info
    
    def compute_reward(self, n_vel, dead):
        reward = 0
        for landmark in self.landmarks:
            dists = [np.linalg.norm(self.drone_pos(drone) -\
                    landmark.state.p_pos) for drone in self.drones]
            # print(dists)
            reward -= min(dists)
        if dead:
            reward -= 50
        # speed = [np.linalg.norm(i) for i in n_vel]
        # reward -= sum(i < speed_limit for i in speed) * 1.0
        return reward

    def interpret_action(self, actions):
        scaling_factor = 5.0
        quad_offset = []
        for action in actions:
            logi_action = np.argmax(action)
            if logi_action == 0:
                curr_offset = (0, 0, 0)
            elif logi_action == 1:
                curr_offset = (scaling_factor, 0, 0)
            elif logi_action == 2:
                curr_offset = (-scaling_factor, 0, 0)
            elif logi_action == 3:
                curr_offset = (0, scaling_factor, 0)    
            elif logi_action == 4:
                curr_offset = (0, -scaling_factor, 0)
            quad_offset.append(curr_offset)
        return quad_offset

    def disconnect(self):
        for drone in self.drones:
            self.client.enableApiControl(False, vehicle_name=drone)
            self.client.armDisarm(False, vehicle_name=drone)
        print("Disconnected.")


if __name__ == "__main__":
    env = Env()
    obs = env.reset()
    quad_offset = [[2,2],[1,1],[3,3]]
    obs, rew, doen, info = env.step(quad_offset)

    # airsim.wait_key('Press any key to reset to original state')
    env.client.armDisarm(False, "Drone1")
    env.client.armDisarm(False, "Drone2")
    env.client.armDisarm(False, "Drone3")
    env.client.reset()
    env.client.enableApiControl(False, "Drone1")
    env.client.enableApiControl(False, "Drone2")
    env.client.enableApiControl(False, "Drone3")
