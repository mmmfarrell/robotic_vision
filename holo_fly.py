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
from simple_pid import PID

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

        # Init previous img
        self.prev_frame = None

        # Optical flow stuff.
        # Make grid of points to compute OF at.
        grid = np.indices((512,512))
        pixels_to_skip = 8;
        row_start = 128
        row_end = 384

        # Make points for left section.
        left = grid[:, row_start:row_end:pixels_to_skip, pixels_to_skip:128:pixels_to_skip]
        left_px = left[1, :, :].reshape((-1,))
        left_py = left[0, :, :].reshape((-1,))
        left_pixels = np.vstack((left_px, left_py))
        self.left_pixels = left_pixels.T.reshape(-1,1,2)

        # Make points for right section.
        right = grid[:, row_start:row_end:pixels_to_skip, 384+pixels_to_skip:512 :pixels_to_skip]
        right_px = right[1, :, :].reshape((-1,))
        right_py = right[0, :, :].reshape((-1,))
        right_pixels = np.vstack((right_px, right_py))
        self.right_pixels = right_pixels.T.reshape(-1,1,2)

        # Things to control off
        self.lr_diff = 0

        # PID controllers. 
        self.vx_pid = PID(p=1.0, d=0.1, max_=np.pi*30./180., min_=-np.pi*30./180.)
        self.vy_pid = PID(p=1.0, d=0.1, max_=np.pi*30./180., min_=-np.pi*30./180.)
        self.PID_yaw = PID(p=1.0, i=0.1)
        self.yaw_control = 0.0
        self.yaw_command = 0.0

        # Flag for automode
        self.autonomous = False

        # Run til we press escape.
        self.run()

    def run(self):
        while(1):
            # Get commands from vimfly
            command, auto, quit, reset = self.vfly.manual_control()

            # Switch autonomous flag
            if auto:
                self.autonomous = not self.autonomous
                print ("Autonomous mode now: ", self.autonomous)

            # Take a step in the holodeck.
            if not self.autonomous:
                state, reward, terminal, _ = self.env.step(command)
            else:
                auto_command = self.auto_control(command)
                state, reward, terminal, _ = self.env.step(auto_command)

            # Process camera and other sensors.
            self.process_sensors(state)
            pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
            if self.prev_frame is None:
                self.prev_frame = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            else:
                self.image_filters(pixels)
                self.prev_frame = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

            # Reset environment if we pressed r
            if reset:
                self.env.reset()

            # Quit if we press ESC
            ch = cv2.waitKey(1)
            if ch == 27 or quit:
                break

    def auto_control(self, command):
        print ("\nCommand in", command)
        auto_command = command.copy()

        # Vx
        vx_desired = 5.0
        auto_command[1] = -self.vx_pid.computePID(vx_desired, self.states[6], 1./30)
        # auto_command[1] = -self.vx_pid.computePID(command[1].copy()*10.0, self.states[6], 1./30)

        # Use OF to compute vy desired
        k = 2.0
        vy_desired = k*self.lr_diff
        print ("vy desired: ", vy_desired)
        auto_command[0] = -self.vy_pid.computePID(vy_desired, self.states[7], 1./30)
        # auto_command[0] = -self.vy_pid.computePID(command[0].copy()*10.0, self.states[7], 1./30)
        # print ("vx desired", command[1].copy()*5.0)
        print ("vx true", self.states[6])
        # print ("vy desired", command[0].copy()*5.0)
        print ("vy true", self.states[7])
        # auto_command[2] = self.PID_yaw.computePID(0., self.states[5], 1./30)
        # print ("auto_command", auto_command)
        return auto_command

    def image_filters(self, pixels):
        """Display camera image in a window. """
        # Get grayscale of new img.
        gray_img = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

        # # Calulate the optical flow at our grid of points.
        # grid_points = (256, 256)
        # points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, grid_points, None)
        # print ("points", points)

        # Example
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Compute left OF.
        p1_left, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.left_pixels.astype(np.float32), None, **lk_params)
        diff_left = p1_left - self.left_pixels
        of_left = np.sqrt(diff_left[:,:,1]**2 + diff_left[:,:,0]**2)
        of_left = np.mean(of_left)

        # Compute right OF.
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.right_pixels.astype(np.float32), None, **lk_params)
        diff = p1 - self.right_pixels
        of = np.sqrt(diff[:,:,1]**2 + diff[:,:,0]**2)
        of_right = np.mean(of)

        # Print difference of optical flows
        # print ("of left: ", of_left)
        # print ("of right: ",  of_right)
        self.lr_diff = (of_left - of_right)/(of_left + of_right)

        # Draw circles at points we are compute OF
        for x, y in self.left_pixels.reshape(-1,2):
            cv2.circle(pixels, (x, y), 2, (0, 255, 0), -1)
        for x, y in self.right_pixels.reshape(-1,2):
            cv2.circle(pixels, (x, y), 2, (0, 255, 0), -1)

        # Draw rectangles
        cv2.rectangle(pixels, (512, 128), (384, 384), (0, 255, 0), 1)
        cv2.rectangle(pixels, (0, 128), (128, 384), (0, 255, 0), 1)

        # Display image in its own window.
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

        # Rotate velocities to get vx, vy
        psi_rot = tf.euler_matrix(0, 0, rpy[2])
        rotated_vel = (psi_rot[0:3,0:3].dot(vel))/100.

        # IMU Sensor
        imu = state[Sensors.IMU_SENSOR]

        # Save off states in array.
        self.states[0:3] = np.resize(position, (3,))
        self.states[3:6] = np.resize(rpy, (3,))
        self.states[6:9] = np.resize(rotated_vel, (3,))
        self.states[9:12] = np.resize(imu[3:6], (3,))
        
def mod2pi(angle):
    while angle < -pi:
        angle += 2*pi
    while angle >= pi:
        angle -= 2*pi
    return angle

if __name__ == '__main__':
    teleop = HoloFly()
