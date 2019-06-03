#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import copy
import math

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._previous_timestamp = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._previous_waypoint  = waypoints[0]
        self._current_waypoint   = waypoints[1]
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self._kp                 = 0.2
        self._kd                 = 0.05
        self._ki                 = 0.01
        self._stanleygain        = 2.5

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
            self._current_waypoint = self._waypoints[min_idx]
        else:
            desired_speed = self._waypoints[-1][2]
            self._current_waypoint = self._waypoints[-1]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('timestep_no',0)
        self.vars.create_var('integral',0.0)
        self.vars.create_var('previous_error',0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            error = (v_desired-v)
            prop = self._kp*error
            self.vars.integral = self.vars.integral + error*(t-self._previous_timestamp)
            integral = self._ki*self.vars.integral
            derivative = self._kd*(error-self.vars.previous_error)/(t-self._previous_timestamp)
            final = prop+integral+derivative
            self.vars.previous_error = error
            self._previous_timestamp = t
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            if final>1:
                final = 1
            if final<-1:
                final=-1
            if(final>0):
                throttle_output = final
                brake_output    = 0
            else:
                throttle_output = 0
                brake_output    = -final

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            # Change the steer output with the lateral controller.
            #print(waypoints[-1])
            #print(waypoints[0][0],waypoints[0][1])
            if(self.vars.timestep_no==0):
                steer_output = 0
                self.vars.timestep_no += 1
            else:
                if(self.vars.timestep_no>=len(waypoints)):
                    min_idx       = 0
                    min_dist      = float("inf")
                    desired_speed = 0
                    for i in range(len(self._waypoints)):
                        dist = np.linalg.norm(np.array([
                                self._waypoints[i][0] - self._current_x,
                                self._waypoints[i][1] - self._current_y]))
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                    if min_idx < len(self._waypoints)-1:

                        self.vars.timestep_no = min_idx+100
                        self._current_waypoint = self._waypoints[self.vars.timestep_no]
                        print(self._current_waypoint)
                        self._previous_waypoint = self._waypoints[self.vars.timestep_no-1]
                    else:
                        self.vars.timestep_no = len(self._waypoints)-1
                else:
                    self._current_waypoint = waypoints[self.vars.timestep_no]
                    self._previous_waypoint = waypoints[self.vars.timestep_no-1]
                print(self.vars.timestep_no)
                print(self._current_waypoint)
                a = self._current_waypoint[1] - self._previous_waypoint[1]
                b = self._previous_waypoint[0] - self._current_waypoint[0]
                c = -a*self._previous_waypoint[0]-b*self._previous_waypoint[1]
                crosstrack_error = (a*x+b*y+c)/(np.sqrt(a*a+b*b))
                crosstrack_steering = np.arctan(self._stanleygain*crosstrack_error/(v+1e-7))
                if(math.isnan(crosstrack_steering)):
                    if (crosstrack_error>0):# and v>0) or (crosstrack_error<0 and v<0):
                        crosstrack_steering = self._pi/2
                    else:
                        crosstrack_steering = -self._pi/2    
                yaw_term = 0
                if self._current_yaw>-self._pi/2 and self._current_yaw<=0:
                    yaw_term = -self._current_yaw
                elif self._current_yaw>-self._pi and self._current_yaw<-self._pi/2:
                     yaw_term = -self._current_yaw
                elif self._current_yaw>0 and self._current_yaw<self._pi/2:
                     yaw_term = self._2pi-self._current_yaw
                elif self._current_yaw>self._pi/2 and self._current_yaw<self._pi:
                     yaw_term = self._2pi-self._current_yaw
                slope = 0
                #print(-a/b,np.isnan(-a/b))
                if np.isnan(-a/b):
                    #print("IM HERE",self._current_waypoint[1],self._previous_waypoint[1])
                    if(a<0):

                        slope = self._pi/2
                    else:
                        slope = 3*self._pi/2
                else:
                    if((-a/b)>0):
                        if(a>0 and -b>0):
                            slope = self._2pi - np.arctan(-a/b)
                        else:
                            
                            slope = self._pi - np.arctan(-a/b)
                    elif (-a/b)<0:
                        if(a>0 and -b<0):
                            slope = self._pi - np.arctan(-a/b)
                        else:
                            slope = -np.arctan(-a/b)
                    elif((-a/b)==0):
                        if -b<0:
                            slope = 0
                        else:
                            slope = self._pi

                
                phi = -((slope)-(yaw_term))
                print(yaw_term,phi)
                if phi>self._pi:
                    phi = -(self._2pi - phi)
                elif phi<-self._pi:
                    phi = self._2pi + phi
                steer_output = crosstrack_steering+phi
                #print(phi,yaw_term,slope)
                #print(crosstrack_steering,phi,steer_output)
                #print(previous_waypoint[0],current_waypoint[0],self.vars.timestep_no,waypoints[641][0],waypoints[642][0])
                if(steer_output<-1.22):
                    #print(self.vars.timestep_no)
                    steer_output = -1.22
                elif steer_output>1.22:
                    #print(self.vars.timestep_no)
                    steer_output = 1.22
                self.vars.timestep_no += 1
                self._previous_waypoint = copy.deepcopy(self._current_waypoint)
                print(yaw_term,phi)
                #steer_output = 1.22
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)
            #print(throttle_output,steer_output,brake_output)
        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
