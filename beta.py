
import numpy as np
import torch
import argparse
from torch import Tensor
from torch import autograd
from torch.autograd import Variable
import sys
print(sys.executable)
print(torch.cuda.is_available())
# import airsim
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True, "Drone1")
# client.enableApiControl(True, "Drone2")
# client.armDisarm(True, "Drone1")
# client.armDisarm(True, "Drone2")

# raise Exception("valid")

# # a = client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
# # b = client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
# f1 = client.takeoffAsync(vehicle_name="Drone1")
# f2 = client.takeoffAsync(vehicle_name="Drone2")
# f1.join()
# f2.join()
# f1 = client.moveToPositionAsync(5, 5, -10, 5, vehicle_name="Drone1")
# f2 = client.moveToPositionAsync(5, 5, -10, 5, vehicle_name="Drone2")
# f1.join()
# f2.join()

# a = client.getGpsData(vehicle_name="Drone1")
# b = client.getMultirotorState(vehicle_name="Drone2").gps_location.latitude

# print(a, b)
# airsim.wait_key('Press any key to reset to original state')

# client.armDisarm(False, "Drone1")
# client.armDisarm(False, "Drone2")
# client.reset()

# # that's enough fun for now. let's quit cleanly
# client.enableApiControl(False, "Drone1")
# client.enableApiControl(False, "Drone2")