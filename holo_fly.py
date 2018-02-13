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

        # Canyon following
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

        # Obstacle avoidance
        # Make points for mleft section.
        mleft = grid[:, row_start:row_end:pixels_to_skip, 64 + pixels_to_skip:256:pixels_to_skip]
        mleft_px = mleft[1, :, :].reshape((-1,))
        mleft_py = mleft[0, :, :].reshape((-1,))
        mleft_pixels = np.vstack((mleft_px, mleft_py))
        self.mleft_pixels = mleft_pixels.T.reshape(-1,1,2)

        # Make points for mright section.
        mright = grid[:, row_start:row_end:pixels_to_skip, 256+pixels_to_skip:448:pixels_to_skip]
        mright_px = mright[1, :, :].reshape((-1,))
        mright_py = mright[0, :, :].reshape((-1,))
        mright_pixels = np.vstack((mright_px, mright_py))
        self.mright_pixels = mright_pixels.T.reshape(-1,1,2)

        # Things to control off
        self.mlr_diff = 0

        # Height
        # Make points for mleft section.
        bot = grid[:, 384:448:pixels_to_skip, 192:321:pixels_to_skip]
        bot_px = bot[1, :, :].reshape((-1,))
        bot_py = bot[0, :, :].reshape((-1,))
        bot_pixels = np.vstack((bot_px, bot_py))
        self.bot_pixels = bot_pixels.T.reshape(-1,1,2)

        # Things to control off
        self.bot = 0
        self.alt_command = 0

        # Time to Collision
        # Make points for mleft section.
        collision_points = grid[:, 200, 200-24:256+24:pixels_to_skip]
        # print ("col points: ", collision_points)
        col_px = collision_points[1, :].reshape((-1,))
        col_py = collision_points[0, :].reshape((-1,))
        col_pixels = np.vstack((col_px, col_py))
        self.col_pixels = col_pixels.T.reshape(-1,2)

        # Things to control off
        self.ttc = 9999

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

                # Set my yaw command
                self.yaw_command = self.states[5]
                self.alt_command = 0.

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
        # print ("\nCommand in", command)
        auto_command = command.copy()

        # Vx
        if self.ttc < 0.25:
            vx_desired = 0.0
            print ("\n\n\n\nSTOP\n\n\n\n")
        else:
            vx_desired = 10.0
        auto_command[1] = -self.vx_pid.computePID(vx_desired, self.states[6], 1./30)
        # auto_command[1] = -self.vx_pid.computePID(command[1].copy()*10.0, self.states[6], 1./30)

        # Use OF to compute vy desired
        k = 2.0
        vy_desired = k*self.lr_diff
        # print ("vy desired: ", vy_desired)
        auto_command[0] = -self.vy_pid.computePID(vy_desired, self.states[7], 1./30)
        # auto_command[0] = -self.vy_pid.computePID(command[0].copy()*10.0, self.states[7], 1./30)
        # print ("vx desired", command[1].copy()*5.0)
        # print ("vx true", self.states[6])
        # print ("vy desired", command[0].copy()*5.0)
        # print ("vy true", self.states[7])
        # auto_command[2] = self.PID_yaw.computePID(0., self.states[5], 1./30)
        # print ("auto_command", auto_command)

        # # Use middle pixels to control yaw
        # if abs(self.mlr_diff) > 0.2:
        #     auto_command[2] = 10.0*self.mlr_diff
        #     # print ("control yaw")
        # else:
        #     auto_command[2] = 0.0
        #     # print ("no")

        # Yaw control
        auto_command[2] = self.PID_yaw.computePID(self.yaw_command, self.states[5], 1./30)

        # Altitude control
        # k_alt = 1.0
        # hdot = k_alt*(5. - self.bot_diff)
        # self.alt_command += (1./30) * hdot
        # auto_command[3] = command[3] + self.alt_command
        # print ("altitude command: ", auto_command[3])

        return auto_command

    def image_filters(self, pixels):
        """Display camera image in a window. """
        # Get grayscale of new img.
        gray_img = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

        # # Calulate the optical flow at our grid of points.
        # grid_points = (256, 256)
        # points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, grid_points, None)
        # print ("points", points)

        pixels = self.compute_lr_diff(pixels, gray_img)
        pixels = self.compute_mlr_diff(pixels, gray_img)
        pixels = self.compute_bot_diff(pixels, gray_img)
        pixels = self.compute_ttc(pixels, gray_img)

        # Print command on screen
        # if self.lr_diff > 0:
        #     print ("Move Right!")
        # else:
        #     print ("Move Left!")
        # if self.mlr_diff > 0:
        #     print ("Yaw Right!")
        # else:
        #     print ("Yaw Left!")

        # Draw circles at points we are compute OF
        for x, y in self.mleft_pixels.reshape(-1,2):
            cv2.circle(pixels, (x, y), 2, (255, 0, 0), -1)
        for x, y in self.mright_pixels.reshape(-1,2):
            cv2.circle(pixels, (x, y), 2, (255, 0, 0), -1)
        for x, y in self.bot_pixels.reshape(-1,2):
            cv2.circle(pixels, (x, y), 2, (0, 0, 255), -1)

        # Draw rectangles
        cv2.rectangle(pixels, (512, 128), (384, 384), (0, 255, 0), 1)
        cv2.rectangle(pixels, (0, 128), (128, 384), (0, 255, 0), 1)


        # Display image in its own window.
        cv2.imshow('Holodeck Image', pixels)

    def compute_lr_diff(self, pixels, gray_img):
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
        left_px_mean = np.sum(diff_left[:,0,0])/len(diff_left[:,0,0])
        left_py_mean = np.sum(diff_left[:,0,1])/len(diff_left[:,0,0])
        # print ("\nLeft means: \n", left_px_mean, "\n", left_py_mean)


        # Compute right OF.
        p1_right, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.right_pixels.astype(np.float32), None, **lk_params)
        diff_right = p1_right - self.right_pixels
        right_px_mean = np.sum(diff_right[:,0,0])/len(diff_right[:,0,0])
        right_py_mean = np.sum(diff_right[:,0,1])/len(diff_right[:,0,0])
        # print ("\nright means: \n", right_px_mean, "\n", right_py_mean)

        # Compute difference of OF
        right_mean_mag = np.sqrt(right_px_mean**2 + right_py_mean**2)
        left_mean_mag = np.sqrt(left_px_mean**2 + left_py_mean**2)
        self.lr_diff = (left_mean_mag - right_mean_mag)/(left_mean_mag + right_mean_mag)
        # self.lr_diff = (left_px_mean - right_px_mean)/(left_px_mean + right_px_mean)
        # print ("LR diff: ", self.lr_diff)

        # Draw OF vector
        scale = 10.
        cv2.arrowedLine(pixels, (64, 256), (int(64 + scale*left_px_mean), int(256 + scale*left_py_mean)), (0, 255, 0), 2)
        cv2.arrowedLine(pixels, (512-64, 256), (int(512 - 64 + scale*right_px_mean), int(256 + scale*right_py_mean)), (0, 255, 0), 2)

        return pixels

    def compute_mlr_diff(self, pixels, gray_img):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Compute left OF.
        p1_left, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.mleft_pixels.astype(np.float32), None, **lk_params)
        diff_left = p1_left - self.mleft_pixels
        left_px_mean = np.sum(diff_left[:,0,0])/len(diff_left[:,0,0])
        left_py_mean = np.sum(diff_left[:,0,1])/len(diff_left[:,0,0])
        # print ("\nLeft means: \n", left_px_mean, "\n", left_py_mean)


        # Compute right OF.
        p1_right, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.mright_pixels.astype(np.float32), None, **lk_params)
        diff_right = p1_right - self.mright_pixels
        right_px_mean = np.sum(diff_right[:,0,0])/len(diff_right[:,0,0])
        right_py_mean = np.sum(diff_right[:,0,1])/len(diff_right[:,0,0])
        # print ("\nright means: \n", right_px_mean, "\n", right_py_mean)

        # Compute difference of OF
        right_mean_mag = np.sqrt(right_px_mean**2 + right_py_mean**2)
        left_mean_mag = np.sqrt(left_px_mean**2 + left_py_mean**2)
        # self.lr_diff = (left_mean_mag - right_mean_mag)/(left_mean_mag + right_mean_mag)
        self.mlr_diff = (abs(left_px_mean) - abs(right_px_mean))/(abs(left_px_mean) + abs(right_px_mean))
        # print ("LR diff: ", self.lr_diff)

        # Draw OF vector
        scale = 10.
        cv2.arrowedLine(pixels, (192, 256), (int(192 + scale*left_px_mean), int(256 + scale*left_py_mean)), (255, 0, 0), 2)
        cv2.arrowedLine(pixels, (256+64, 256), (int(256 + 64 + scale*right_px_mean), int(256 + scale*right_py_mean)), (255, 0, 0), 2)

        return pixels

    def compute_bot_diff(self, pixels, gray_img):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Compute left OF.
        p1_bot, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.bot_pixels.astype(np.float32), None, **lk_params)
        diff_bot = p1_bot - self.bot_pixels
        bot_px_mean = np.sum(diff_bot[:,0,0])/len(diff_bot[:,0,0])
        bot_py_mean = np.sum(diff_bot[:,0,1])/len(diff_bot[:,0,0])
        # print ("\nLeft means: \n", left_px_mean, "\n", left_py_mean)


        # # Compute difference of OF
        bot_mean_mag = np.sqrt(bot_px_mean**2 + bot_py_mean**2)
        # left_mean_mag = np.sqrt(left_px_mean**2 + left_py_mean**2)
        # # self.lr_diff = (left_mean_mag - right_mean_mag)/(left_mean_mag + right_mean_mag)
        # self.mlr_diff = (abs(left_px_mean) - abs(right_px_mean))/(abs(left_px_mean) + abs(right_px_mean))
        self.bot_diff = bot_mean_mag
        # print ("Bot y: ", bot_mean_mag)
        #
        # # Draw OF vector
        scale = 10.
        cv2.arrowedLine(pixels, (256, 384+64), (int(256 + scale*bot_px_mean), int(384 + 64 + scale*bot_py_mean)), (0, 0, 255), 2)
        # cv2.arrowedLine(pixels, (256+64, 256), (int(256 + 64 + scale*right_px_mean), int(256 + scale*right_py_mean)), (255, 0, 0), 2)

        return pixels

    def compute_ttc(self, pixels, gray_img):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Compute left OF.
        p1_col, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_img, self.col_pixels.astype(np.float32), None, **lk_params)
        diff_col = p1_col - self.col_pixels
        # print ("ttc: ", diff_col)
        # print ("ttc pix: ", self.col_pixels)

        sx = (self.col_pixels[:,0] - 255)
        sxdot = diff_col[:,0]*30.
        ttc = abs(np.divide(sx, sxdot))
        # print ("sx: ",sx)
        # print ("sx dot: ", sxdot)
        print ("ttc: ", ttc.mean())
        self.ttc = ttc.mean()

        # col_px_mean = np.sum(diff_col[:,0,0])/len(diff_col[:,0,0])
        # # bot_py_mean = np.sum(diff_bot[:,0,1])/len(diff_bot[:,0,0])
        # # print ("\nLeft means: \n", left_px_mean, "\n", left_py_mean)
        #
        #
        # # # Compute difference of OF
        # bot_mean_mag = np.sqrt(bot_px_mean**2 + bot_py_mean**2)
        # # left_mean_mag = np.sqrt(left_px_mean**2 + left_py_mean**2)
        # # # self.lr_diff = (left_mean_mag - right_mean_mag)/(left_mean_mag + right_mean_mag)
        # # self.mlr_diff = (abs(left_px_mean) - abs(right_px_mean))/(abs(left_px_mean) + abs(right_px_mean))
        # self.bot_diff = bot_mean_mag
        # print ("Bot y: ", bot_mean_mag)
        # #
        # # # Draw OF vector
        # scale = 10.
        # cv2.arrowedLine(pixels, (256, 384+64), (int(256 + scale*bot_px_mean), int(384 + 64 + scale*bot_py_mean)), (0, 0, 255), 2)
        # cv2.arrowedLine(pixels, (256+64, 256), (int(256 + 64 + scale*right_px_mean), int(256 + scale*right_py_mean)), (255, 0, 0), 2)

        return pixels

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
