#!/usr/bin/env python
"""
Script to fly holodeck with your keyboard.

Note: VimFly window must be selected for keys
to register. See VimFly for documentation on keys
for flight.

"""
import numpy as np
import time
import cv2
import math

from fly import VimFly
import transformations as tf

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

class HoloFly(object):
    """Docstring for HoloFly. """

    def __init__(self):
        """TODO: to be defined1. """
        # Init an instance of holodeck environment.
        self.env = Holodeck.make("UrbanCity")

        # Init vimfly to get keyboard commands.
        self.vfly = VimFly()

        # Init our state vector:
        # x = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        self.states = np.zeros((12,))
        self.commands = np.zeros((4,))

        # Make an opencv window.
        cv2.namedWindow('Holodeck Image')

        # Run til we press escape.
        self.run()

    def run(self):
        while(1):
            # Get commands from vimfly
            command, auto, quit, reset = self.vfly.manual_control()

            # Take a step in the holodeck.
            state, reward, terminal, _ = self.env.step(command)

            # Process camera and other sensors.
            self.process_sensors(state)
            pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
            self.image_filters(pixels)

            # Reset environment if we pressed r
            if reset:
                self.env.reset()

            # Quit if we press ESC
            ch = cv2.waitKey(1)
            if ch == 27 or quit:
                break

    def image_filters(self, pixels):
        """Display camera image in a window. """
        cv2.imshow('Holodeck Image', pixels)

    def process_sensors(self, state):
        """Put sensor data into state vector. """
        # Orientation Sensor
        orientation = state[Sensors.ORIENTATION_SENSOR]

        # Make orientation 4x4 homogenous matrix.
        homo_orientation = np.diag([1., 1., 1., 1.])
        homo_orientation[0:3, 0:3] = orientation

        # Convert homo_orientation to roll, pitch, yaw as we know.
        rpy = tf.euler_from_matrix(homo_orientation, axes='rxyz')

        # Position Sensor
        position = state[Sensors.LOCATION_SENSOR]

        # Velocity Sensor
        vel = state[Sensors.VELOCITY_SENSOR]

        # IMU Sensor
        imu = state[Sensors.IMU_SENSOR]

        # Save off states in array.
        self.states[0:3] = np.resize(position, (3,))
        self.states[3:6] = np.resize(rpy, (3,))
        self.states[6:9] = np.resize(vel, (3,))
        self.states[9:12] = np.resize(imu[3:6], (3,))
        
def mod2pi(angle):
    while angle < -pi:
        angle += 2*pi
    while angle >= pi:
        angle -= 2*pi
    return angle

if __name__ == '__main__':
    teleop = HoloFly()
