#!/usr/bin/env python
"""
vimfly - vim keybindings for your multirotor!

Teleoperated flying from your keyboard. Command u, v, psidot, and altitude.

The following keybindings are used:
    - a: lower altitude
    - s: higher altitude
    - d: CCW (-psidot)
    - f: CW (+psidot)
    - h: Left (-v)
    - j: Backward (-u)
    - k: Forward (+u)
    - l: Right (+v)

Connect to rosflight_io command topic (SIL/HW) or roscopter_sim Gazebo command
topic.


https://xkcd.com/1823/
"""
import pygame
import numpy as np
import time
import cv2
import math

class VimFly:
    def __init__(self):

        # initialize pygame display
        pygame.init()
        pygame.display.set_caption('vimfly')
        self.screen = pygame.display.set_mode((550, 200))
        self.font = pygame.font.SysFont("monospace", 18)

        # retrieve vimfly parameters from the rosparam server
        self.params = {
            'roll_cmd': 0.5,
            'pitch_cmd': 0.5,
            'psidot_cmd': 3.14159,
            'altitude_cmd': 10.0, # start at this altitude
            'altitude_step': 0.1, # each step inc/dec by this amount
        }

        # Init our altitude command.
        self.altitude = self.params['altitude_cmd']

        # Allow continuous holding of altitude keys without growing too big
        self.alt_debouncing = False
        self.ALT_DEBOUNCE_THRESHOLD = 0.100

    def manual_control(self):
            keys = pygame.key.get_pressed()

            # Init constant for quit, reset, and autonomous control
            quit = False
            auto = False
            reset = False

            # Escape button to quit
            if keys[pygame.K_ESCAPE]:
                quit = True

            # Press r to reset the environment.
            if keys[pygame.K_r]:
                reset = True

                # Also reset the altitude command
                self.altitude = self.params['altitude_cmd']

            # Roll commands, Left = h, Right = l
            if keys[pygame.K_h]:
                roll = self.params['roll_cmd']
            elif keys[pygame.K_l]:
                roll = -self.params['roll_cmd']
            else:
                roll = 0

            # Pitch commands, Forward = k, backward = j 
            if keys[pygame.K_k]:
                pitch  = -self.params['pitch_cmd']
            elif keys[pygame.K_j]:
                pitch  = self.params['pitch_cmd']
            else:
                pitch = 0

            # Yaw rate commands, CCW = d, CW = f
            if keys[pygame.K_d]:
                r = self.params['psidot_cmd']
            elif keys[pygame.K_f]:
                r = -self.params['psidot_cmd']
            else:
                r = 0

            # Altitude commands, lower = a, higher = s
            if keys[pygame.K_a] or keys[pygame.K_s] or keys[pygame.K_y]:
                if not self.alt_debouncing:
                    self.alt_debouncing = True
                    self.alt_start_time = time.time()

                if (time.time() - self.alt_start_time) > self.ALT_DEBOUNCE_THRESHOLD:
                    # The key has been debounced once, start the process over!
                    self.alt_debouncing = False

                    # Increment the commanded altitude
                    if keys[pygame.K_a]:
                        self.altitude -= self.params['altitude_step']

                    elif keys[pygame.K_s]:
                        self.altitude += self.params['altitude_step']

                    # Enter autonomous mode (flag)
                    if keys[pygame.K_y]:
                        auto = True

            else:
                self.alt_debouncing = False

            # Command to output
            command = np.array([roll, pitch, r, self.altitude])

            # Update the display with the current commands
            self.update_display(command)

            # process event queue and throttle the while loop
            pygame.event.pump()

            # Return our command with auto/quit flags
            return command, auto, quit, reset


    def update_display(self, command):
        self.display_help()

        msgText = "roll: {}, pitch: {}, psidot: {}, alt: {}".format(command[0], command[1], command[2], command[3])
        self.render(msgText, (0,140))

        pygame.display.flip()


    def display_help(self):
        self.screen.fill((0,0,0))

        LINE=20

        self.render("vimfly keybindings:", (0,0))
        self.render("- a: lower altitude", (0,1*LINE)); self.render("- h: Left (-v)", (250,1*LINE))
        self.render("- s: higher altitude", (0,2*LINE)); self.render("- j: Backward (-u)", (250,2*LINE))
        self.render("- d: CCW (-psidot)", (0,3*LINE)); self.render("- k: Forward (+u)", (250,3*LINE))
        self.render("- f: CW (+psidot)", (0,4*LINE)); self.render("- l: Right (+v)", (250,4*LINE))


    def render(self, text, loc):
        txt = self.font.render(text, 1, (255,255,255))
        self.screen.blit(txt, loc)

if __name__ == '__main__':
    teleop = VimFly()
