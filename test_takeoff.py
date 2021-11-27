import airsim
import time
import os
import tempfile
import cv2
import argparse
import numpy as np



# connect to the AirSim simulator
class Env:
    def __init__(self):
        """
        Connect to the AirSim simulator.
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.num_drones = 3
        self.ctrl_list = []  # control buffer
        # serial number of drones
        self.drones = [('Drone' + str(i+1)) for i in range(self.num_drones)]

    def reset(self):
        self.client.reset()
        for drone in self.drones:
            self.client.enableApiControl(True, drone)

        # begin simulate
        self.client.simPause(False)

        # drones takeoff
        for drone in self.drones:
            self.ctrl_list.append(
                self.client.moveByVelocityZAsync(0, 0, -1, 4*args.time_slice,
                                                vehicle_name=drone))
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()

        # drones cushion
        for drone in self.drones:
            self.ctrl_list.append(
                self.client.moveByVelocityAsync(0, 0, 0, 0.1*args.time_slice,
                                                vehicle_name=drone))
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()

        # drones takeoff
        for drone in self.drones:
            self.ctrl_list.append(
                self.client.moveByVelocityZAsync(0, 1, -1, 4*args.time_slice,
                                                vehicle_name=drone))
        for ctrl in self.ctrl_list:
            ctrl.join()
        self.ctrl_list.clear()

        # # drones hover
        # for drone in self.drones:
        #     self.ctrl_list.append(self.client.hoverAsync(vehicle_name=drone))
        # for ctrl in self.ctrl_list:
        #     ctrl.join()
        # self.ctrl_list.clear()
        
        # # pause simulate
        # self.client.simPause(True)

        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        # quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])

        # i = 1



# for drone in drones:
#     client.takeoffAsync(vehicle_name=drone)    # takeoff
# time.sleep(10)
# client.hoverAsync().join()

# for drone in drones:
#     client.moveByVelocityAsync(0, 0, -1, 4*time_slice,
#                                     vehicle_name=drone)
# client.moveByVelocityAsync(3, 3, -3, 3).join()
# client.moveByVelocityAsync(0, 3, -3, 3).join()


# responses1 = client.simGetImages([
#     airsim.ImageRequest("4", airsim.ImageType.DepthVis),  #depth visualization image
#     airsim.ImageRequest("4", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
# print('Drone: Retrieved images: %d' % len(responses1))

# tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
# print ("Saving images to %s" % tmp_dir)
# try:
#     os.makedirs(tmp_dir)
# except OSError:
#     if not os.path.isdir(tmp_dir):
#         raise

# for idx, response in enumerate(responses1):

#     filename = os.path.join(tmp_dir, str(idx))

#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#     elif response.compress: #png format
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#     else: #uncompressed array
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
#         img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
#         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


# for drone in drones:
#     client.landAsync(vehicle_name=drone)       # land

# time.sleep(10)
# client.armDisarm(False)         # lock
# client.enableApiControl(False)  # release control
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_slice", type=float, default=0.5)
    args = parser.parse_args()
    env = Env()
    env.reset()