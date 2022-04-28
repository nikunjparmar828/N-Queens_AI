from cmath import log10
import time
from typing import List, Tuple, Union
import numpy as np
import pygame
import random
import csv
import copy
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import pi

class Node():

    def __init__(self, theta1, theta_dot_1, theta2, theta_dot_2):
        self.theta_1 = theta1
        self.theta_2 = theta2
        self.theta_dot_1 = theta_dot_1
        self.theta_dot_2 = theta_dot_2 

        self.Q_value_positive_torque_1 = 0
        self.Q_value_positive_torque_2 = 0

        self.prev_Q_value_positive_torque_1 = 0
        self.prev_Q_value_positive_torque_2 = 0

        self.Q_value_negative_torque_1 = 0
        self.Q_value_negative_torque_2 = 0

        self.prev_Q_value_negative_torque_1 = 0
        self.prev_Q_value_negative_torque_2 = 0

        self.alpha = 0.05

    def get_Q_value(self, action):
        if action[0] == "positive" and action[1] == "positive":
            return [self.Q_value_positive_torque_1, self.Q_value_positive_torque_2]
        elif action[0] == "positive" and action[1] == "negative":
            return [self.Q_value_positive_torque_1, self.Q_value_negative_torque_2]
        elif action[0] == "negative" and action[1] == "negative":
            return [self.Q_value_negative_torque_1, self.Q_value_negative_torque_2]
        elif action[0] == "negative" and action[1] == "positive":
            return [self.Q_value_negative_torque_1, self.Q_value_positive_torque_2]
            

    def update_Q_value(self, update_value1, update_value2, direction1, direction2):
        if direction1 == "positive" and direction2 == "positive":
            self.prev_Q_value_positive_torque_1 = self.Q_value_positive_torque_1
            self.prev_Q_value_positive_torque_2 = self.Q_value_positive_torque_2
            
            self.Q_value_positive_torque_1 = update_value1
            self.Q_value_positive_torque_2 = update_value2

        elif direction1 == "positive" and direction2 == "negative":
            self.prev_Q_value_positive_torque_1 = self.Q_value_positive_torque_1
            self.prev_Q_value_negative_torque_2 = self.Q_value_negative_torque_2
            
            self.Q_value_positive_torque_1 = update_value1
            self.Q_value_negative_torque_2 = update_value2

        elif direction1 == "negative" and direction2 == "positive":
            self.prev_Q_value_negative_torque_1 = self.Q_value_negative_torque_1
            self.prev_Q_value_positive_torque_2 = self.Q_value_positive_torque_2
            
            self.Q_value_negative_torque_1 = update_value1
            self.Q_value_positive_torque_2 = update_value2
        
        elif direction1 == "negative" and direction2 == "negative":
            self.prev_Q_value_negative_torque_1 = self.Q_value_negative_torque_1
            self.prev_Q_value_negative_torque_2 = self.Q_value_negative_torque_2
            
            self.Q_value_negative_torque_1 = update_value1
            self.Q_value_negative_torque_2 = update_value2



    def get_max_action(self):
        actions1 = [self.Q_value_positive_torque_1, self.Q_value_negative_torque_1]
        actions2 = [ self.Q_value_positive_torque_2,  self.Q_value_negative_torque_2]
        
        action1 = max(actions1)
        action2 = max(actions2)

        index1 = actions1.index(action1)
        index2 = actions2.index(action2)
        
        if index1 == 0 and index2 == 0:
            return ["positive1", "positive2"]
        if index1 == 0 and index2 == 1:
            return ["positive1", "negative2"]
        if index1 == 1 and index2 == 0:
            return ["negative1", "positive2"]
        if index1 == 1 and index2 == 1:
            return ["negative1", "negative2"] 


    def get_force(self, direction1, direction2):
        if direction1 == "positive" and direction2 == "positive":
            delta_Q1 = self.Q_value_positive_torque_1-self.prev_Q_value_positive_torque_1
            delta_Q2 = self.Q_value_positive_torque_2-self.prev_Q_value_positive_torque_2

            if delta_Q1 == 0 and delta_Q2 == 0:
                return [1, 1]
            elif delta_Q1 == 0 and not delta_Q2 == 0:
                return [1, self.alpha * (delta_Q2)]
            elif not delta_Q1 == 0 and delta_Q2 == 0:
                return [self.alpha * (delta_Q1), 1]
            elif not delta_Q1 == 0 and not delta_Q2 == 0:
                return [self.alpha*(delta_Q1), self.alpha*(delta_Q2)]

        elif direction1 == "positive" and direction2 == "negative":
            delta_Q1 = self.Q_value_positive_torque_1-self.prev_Q_value_positive_torque_1
            delta_Q2 = self.Q_value_negative_torque_2-self.prev_Q_value_negative_torque_2

            if delta_Q1 == 0 and delta_Q2 == 0:
                return [1, -1]
            elif delta_Q1 == 0 and not delta_Q2 == 0:
                return [1, self.alpha * (delta_Q2)]
            elif not delta_Q1 == 0 and delta_Q2 == 0:
                return [self.alpha * (delta_Q1), -1]
            elif not delta_Q1 == 0 and not delta_Q2 == 0:
                return [self.alpha*(delta_Q1), self.alpha*(delta_Q2)]

        elif direction1 == "negative" and direction2 == "positive":
            delta_Q1 = self.Q_value_negative_torque_1-self.prev_Q_value_negative_torque_1
            delta_Q2 = self.Q_value_positive_torque_2-self.prev_Q_value_positive_torque_2
            
            if delta_Q1 == 0 and delta_Q2 == 0:
                return [-1, 1]
            elif delta_Q1 == 0 and not delta_Q2 == 0:
                return [-1, self.alpha * (delta_Q2)]
            elif not delta_Q1 == 0 and delta_Q2 == 0:
                return [self.alpha * (delta_Q1), 1]
            elif not delta_Q1 == 0 and not delta_Q2 == 0:
                return [self.alpha*(delta_Q1), self.alpha*(delta_Q2)]

        elif direction1 == "negative" and direction2 == "negative":
            delta_Q1 = self.Q_value_negative_torque_1-self.prev_Q_value_negative_torque_1
            delta_Q2 = self.Q_value_negative_torque_2-self.prev_Q_value_negative_torque_2

            if delta_Q1 == 0 and delta_Q2 == 0:
                return [-1, -1]
            elif delta_Q1 == 0 and not delta_Q2 == 0:
                return [-1, self.alpha * (delta_Q2)]
            elif not delta_Q1 == 0 and delta_Q2 == 0:
                return [self.alpha * (delta_Q1), -1]
            elif not delta_Q1 == 0 and not delta_Q2 == 0:
                return [self.alpha*(delta_Q1), self.alpha*(delta_Q2)]

        # if direction == "negative":
        #     delta_Q = self.Q_value_negative_force - self.prev_Q_value_negative_force
        #     if delta_Q == 0:
        #         return -1
        #     elif not delta_Q == 0:

        #         return self.alpha * (delta_Q)

class world:

    def __init__(self, P_lim1, V_lim1,  P_lim2, V_lim2, step):
        self.P_lim1 = P_lim1
        self.V_lim1 = V_lim1
        self.P_lim2 = P_lim2
        self.V_lim2 = V_lim2

        self.step = step
        self.row_length1 = int((2*P_lim1)/step) # 100
        self.col_length1 = int((2*V_lim1)/step) # 100
        self.row_length2 = int((2*P_lim2)/step) # 100
        self.col_length2 = int((2*V_lim2)/step) # 100

        self.epsilon = 0.1 

        # GRID
        # Assume that this works --> I am not 100% sure if this works 
        self.states = [[[[Node((i*step) - 1, (j*step) - V_lim1, (n*step) - 1, (m*step) - V_lim2) for j in range(self.col_length1)] for i in range(self.row_length1) ] for m in range(self.col_length2)] for n in range(self.row_length2)]

    def simulation_setup(self, time_step, torque1, torque2, mass1, mass2, l1, l2, I1, I2):
        self.time_step = time_step
        self.mass1 = mass1
        self.mass2 = mass2
        self.l1 = l1
        self.l2 = l2
        self.r1 = I1
        self.r2 = I2
        self.torque1 = torque1
        self.torque2 = torque2

    def get_co_ordinates(self, pos1, vel1, pos2, vel2):
        i = (pos1 + self.P_lim1)/self.step
        j = (vel1 + self.V_lim1)/self.step
        m = (pos2 + self.P_lim2)/self.step
        n = (vel2 + self.V_lim2)/self.step
        
        return int(i), int(j), int(m), int(n)

    def define_start_pos(self, pos1, vel1, pos2, vel2):
        i, j, m, n = self.get_co_ordinates(pos1, vel1, pos2, vel2)

        self.start = self.states[i][j][m][n]

    def define_target_pos(self,  pos1, vel1, pos2, vel2):
        i, j, m, n = self.get_co_ordinates(pos1, vel1, pos2, vel2)
        self.target = self.states[i][j][m][n]

    def check_lim(self,  pos1, vel1, pos2, vel2):
        if abs(pos1) > self.P_lim1 or abs(vel1) > self.V_lim1 or abs(pos2) > self.P_lim2 or abs(vel2) > self.V_lim2:
            return False
        else:
            return True

# MOTION MODEL --> update it according to the two link model 
    def motion(self, control_torque1, control_torque2):
        total_torque_1 = self.torque1 + control_torque1
        total_torque_2 = self.torque2 + control_torque2
        
        # print(total_force)
        
        # acc = total_torque_1/self.mass
        # print(acc)
        self.current_vel = self.current_vel + self.time_step*acc
        self.current_pos = self.current_pos + self.current_vel*self.time_step + 0.5*acc*(self.time_step**2)

    def policy_Epsilon_greedy(self, current_node):
        p = np.random.random()
        if p <= self.epsilon:
            return random.choice([["positive1", "positive2"], ["positive1", "negative2"], ["negative1", "negative2"], ["negative1", "positive2"] ])
        elif p > self.epsilon:
            return current_node.get_max_action()

    def learning(self):
        self.alpha = 0.2
        self.gamma = 0.9

        t_init = time.time()
        t_end = time.time()
        iter = 0
        while t_end - t_init <= 30:
            print(iter)
            current_node = self.start
            self.current_pos1 = current_node.pos1
            self.current_vel1 = current_node.vel1
            self.current_pos2 = current_node.pos2
            self.current_vel2 = current_node.vel2


            print(current_node.pos1, current_node.vel1, current_node.pos2, current_node.vel2)

            # RATE OF CHANGE OF ERROR
            prev_error_p1 = self.target.pos1 - self.current_pos1
            prev_error_v1 = self.target.vel1 - self.current_vel1

            prev_error_p2 = self.target.pos2 - self.current_pos2
            prev_error_v2 = self.target.vel2 - self.current_vel2

            while not current_node == self.target and not current_node == None and t_end - t_init <= 30:
                # print(current_node.pos)
                # print(current_node == self.target )
                action = self.policy_Epsilon_greedy(current_node)
                print(action)

                # force = alpha*deltaQ
                force1, force2 = current_node.get_force(action[0], action[1])

                # print(force)
                self.motion(force1, force2)  # take action step
                print(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2)
                # exit(1)

                # AGENT MUST BE INSIDE THE POS AND VEL LIMIT
                if not self.check_lim(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2):
                    reward = -10
                    Q_ns_na = 0
                    Q_s_a = current_node.get_Q_value(action)
                    new_state = None

                elif self.check_lim(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2):
                    i, j, m, n = self.get_co_ordinates(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2)
                    new_state = self.states[i][j][m][n]   # to do: must be within limit, will  give error if not

                    # error term
                    error_p1 = self.target.pos1 - self.current_pos1
                    error_v1 = self.target.vel1 - self.current_vel1
                    error_p2 = self.target.pos2 - self.current_pos2
                    error_v2 = self.target.vel2 - self.current_vel2
                    # if prev_error_p-error_p > 0;
                    #     reward = 0.5
                    # REWARD
                    reward = 1*np.sign(abs(prev_error_p1) - abs(error_p1)) + 0.1*np.sign(abs(prev_error_v1) - abs(error_v1)) + 1*np.sign(abs(prev_error_p2) - abs(error_p2)) + 0.1*np.sign(abs(prev_error_v2) - abs(error_v2))
                    
                    if not np.sign(prev_error_p1) == np.sign(error_p1) or not np.sign(prev_error_p2) == np.sign(error_p2):
                        reward += -10
                    Q_ns_na = max([new_state.get_Q_value(["positive", "positive"]), new_state.get_Q_value(["positive", "negative"]), new_state.get_Q_value(["negative", "negative"]),new_state.get_Q_value(["negative", "positive"]) ])
                    Q_s_a = current_node.get_Q_value(action)

                    if new_state == self.target:
                        reward += 5
                        print("____________________________DONE___________________________")
                        # time.sleep(3)
                    else:
                        reward += -0.1

                # UPDATE STEP
                print(reward)
                update_value1 = Q_s_a[0] + self.alpha*(reward + (self.gamma*Q_ns_na[0]) - Q_s_a[0])
                update_value2 = Q_s_a[1] + self.alpha*(reward + (self.gamma*Q_ns_na[1]) - Q_s_a[1])

                current_node.update_Q_value(update_value1,update_value2, action)
                current_node = new_state
                prev_error_v1 = error_v1
                prev_error_p1 = error_p1
                prev_error_v2 = error_v2
                prev_error_p2 = error_p2

                reward = 0
                t_end = time.time()  # to cancel the sleep time
            print(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2)

            iter += 1

        print("DONE")

        # TESTING LOOP
        time.sleep(5)
        path = []
        current_node = self.start
        
        self.current_pos1 = current_node.pos1
        self.current_vel1 = current_node.vel1
        self.current_pos2 = current_node.pos2
        self.current_vel2 = current_node.vel2

        while not current_node == self.target:
            path.append([current_node.pos1,current_node.pos2])
            action = current_node.get_max_action()
            print(action)
            force1, force1 = current_node.get_force(action)
        
            # print(force)
            self.motion(force1, force2)  # take action step
            print(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2)
            # exit(1)
            if not self.check_lim(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2):
                reward = -10
                Q_ns_na = 0
                Q_s_a = current_node.get_Q_value(action)
                new_state = None
        
            elif self.check_lim(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2):
                i, j, m, n = self.get_co_ordinates(self.current_pos1, self.current_vel1, self.current_pos2, self.current_vel2)
                new_state = self.states[i][j][m][n]  # to do: must be within limit, will  give error if not
                
                
                error_p1 = self.target.pos1 - self.current_pos1
                error_v1 = self.target.vel1 - self.current_vel1
                error_p2 = self.target.pos2 - self.current_pos2
                error_v2 = self.target.vel2 - self.current_vel2
                
                # if prev_error_p-error_p > 0;
                #     reward = 0.5
                # reward = 1 * np.sign(abs(prev_error_p) - abs(error_p)) + 0.1 * np.sign(abs(prev_error_v) - abs(error_v))
                # if not np.sign(prev_error_p) == np.sign(error_p):
                #     reward += -10
                # Q_ns_na = max([new_state.get_Q_value("positive"), new_state.get_Q_value("negative")])
                # Q_s_a = current_node.get_Q_value(action)
        
                reward = 1*np.sign(abs(prev_error_p1) - abs(error_p1)) + 0.1*np.sign(abs(prev_error_v1) - abs(error_v1)) + 1*np.sign(abs(prev_error_p2) - abs(error_p2)) + 0.1*np.sign(abs(prev_error_v2) - abs(error_v2))
                
                if not np.sign(prev_error_p1) == np.sign(error_p1) or not np.sign(prev_error_p2) == np.sign(error_p2):
                    reward += -10
                Q_ns_na = max([new_state.get_Q_value(["positive", "positive"]), new_state.get_Q_value(["positive", "negative"]), new_state.get_Q_value(["negative", "negative"]),new_state.get_Q_value(["negative", "positive"]) ])
                Q_s_a = current_node.get_Q_value(action)

                if new_state == self.target:
                    reward += 5
                    print("____________________________DONE___________________________")
                    time.sleep(3)
                    break
                else:
                    reward += -0.1
        
            # print(reward)
            # update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
            # # print("new", update_value)
            # current_node.update_Q_value(update_value, action)
            # # print(current_node.Q_value_positive_force, current_node.Q_value_negative_force)
            # current_node = new_state
            # prev_error_v = error_v
            # prev_error_p = error_p
            # reward = 0
            # UPDATE STEP
            print(reward)
            update_value1 = Q_s_a[0] + self.alpha*(reward + (self.gamma*Q_ns_na[0]) - Q_s_a[0])
            update_value2 = Q_s_a[1] + self.alpha*(reward + (self.gamma*Q_ns_na[1]) - Q_s_a[1])

            current_node.update_Q_value(update_value1,update_value2, action)
            current_node = new_state
            prev_error_v1 = error_v1
            prev_error_p1 = error_p1
            prev_error_v2 = error_v2
            prev_error_p2 = error_p2

            reward = 0
   


if __name__ == '__main__':
    sim_world = world(5, 5, 5, 5, 0.1)
    sim_world.simulation_setup(0.1, 0, 0, 1, 1, 0.25, 0.25) #(time_step, torque1, torque2, mass1, mass2, l1, l2, I1, I2)
    
    # (pos1, pos2) -> (pi/2,0) -> lying right side 
    # (pos1, pos2) -> (0,0) -> upward position 
    # (pos1, pos2) -> (-pi/2,0) -> lying left side

    sim_world.define_start_pos(pi/2, 0, 0, 0) # pos_1: 0, vel_1: 0, pos_2: 0,vel_2: 0 --> vel in rad/s and pos in rad 
    sim_world.define_target_pos(0, 0, 0, 0)
    sim_world.learning()


# ##------------------------------------------------------------------------------------------------------------------
# #Simulation

# class Robot:
#     JOINT_LIMITS = [-6.28, 6.28]
#     # MAX_VELOCITY = 15 # 15
#     # MAX_ACCELERATION = 50 #50
#     DT = 0.033

#     link_1: float = 75.  # pixels
#     link_2: float = 50.  # pixels
#     _theta_0: float      # radians
#     _theta_1: float      # radians

#     def __init__(self) -> None:
#         # internal variables
#         self.all_theta_0: List[float] = []
#         self.all_theta_1: List[float] = []

#         self.theta_0 = 0.
#         self.theta_1 = 0.

#     # Getters/Setters
#     @property
#     def theta_0(self) -> float:
#         return self._theta_0 
     
#     @theta_0.setter
#     def theta_0(self, value: float) -> None:
#         self.all_theta_0.append(value)
#         self._theta_0 = value
#         # Check limits
#         # assert self.check_angle_limits(value), \
#         #     f'Joint 0 value {value} exceeds joint limits'
#         # assert self.max_velocity(self.all_theta_0) < self.MAX_VELOCITY, \
#         #     f'Joint 0 Velocity {self.max_velocity(self.all_theta_0)} exceeds velocity limit'
#         # assert self.max_acceleration(self.all_theta_0) < self.MAX_ACCELERATION, \
#         #     f'Joint 0 Accel {self.max_acceleration(self.all_theta_0)} exceeds acceleration limit'

#     @property
#     def theta_1(self) -> float:
#         return self._theta_1

#     @theta_1.setter
#     def theta_1(self, value: float) -> None:
#         self.all_theta_1.append(value)
#         self._theta_1 = value
#         # assert self.check_angle_limits(value), \
#         #     f'Joint 1 value {value} exceeds joint limits'
#         # assert self.max_velocity(self.all_theta_1) < self.MAX_VELOCITY, \
#         #     f'Joint 1 Velocity {self.max_velocity(self.all_theta_1)} exceeds velocity limit'
#         # assert self.max_acceleration(self.all_theta_1) < self.MAX_ACCELERATION, \
#         #     f'Joint 1 Accel {self.max_acceleration(self.all_theta_1)} exceeds acceleration limit'

#     # Kinematics
#     def joint_1_pos(self) -> Tuple[float, float]:
#         """
#         Compute the x, y position of joint 1
#         """
#         return self.link_1 * np.cos(self.theta_0), self.link_1 * np.sin(self.theta_0)

#     def joint_2_pos(self) -> Tuple[float, float]:
#         """
#         Compute the x, y position of joint 2
#         """
#         return self.forward(self.theta_0, self.theta_1)

#     @classmethod
#     def forward(cls, theta_0: float, theta_1: float) -> Tuple[float, float]:
#         """
#         Compute the x, y position of the end of the links from the joint angles
#         """
#         x = cls.link_1 * np.cos(theta_0) + cls.link_2 * np.cos(theta_0 + theta_1)
#         y = cls.link_1 * np.sin(theta_0) + cls.link_2 * np.sin(theta_0 + theta_1)

#         return x, y

#     @classmethod
#     def inverse(cls, x: float, y: float) -> Tuple[float, float]:
#         """
#         Compute the joint angles from the position of the end of the links
#         """
#         theta_1 = np.arccos((x ** 2 + y ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)
#                             / (2 * cls.link_1 * cls.link_2))
#         theta_0 = np.arctan2(y, x) - \
#             np.arctan((cls.link_2 * np.sin(theta_1)) /
#                       (cls.link_1 + cls.link_2 * np.cos(theta_1)))

#         return theta_0, theta_1

#     @classmethod
#     def check_angle_limits(cls, theta: float) -> bool:
#         return cls.JOINT_LIMITS[0] < theta < cls.JOINT_LIMITS[1]

#     @classmethod
#     def max_velocity(cls, all_theta: List[float]) -> float:
#         return float(max(abs(np.diff(all_theta) / cls.DT), default=0.))

#     @classmethod
#     def max_acceleration(cls, all_theta: List[float]) -> float:
#         return float(max(abs(np.diff(np.diff(all_theta)) / cls.DT / cls.DT), default=0.))

#     @classmethod
#     def min_reachable_radius(cls) -> float:
#         return max(cls.link_1 - cls.link_2, 0)

#     @classmethod
#     def max_reachable_radius(cls) -> float:
#         return cls.link_1 + cls.link_2


# class World:
#     def __init__(
#         self,
#         width: int,
#         height: int,
#         robot_origin: Tuple[int, int],
#         goal: Tuple[int, int]
#     ) -> None:
#         self.width = width
#         self.height = height
#         self.robot_origin = robot_origin
#         self.goal = goal

#     def convert_to_display(
#             self, point: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
#         """
#         Convert a point from the robot coordinate system to the display coordinate system
#         """
#         robot_x, robot_y = point
#         offset_x, offset_y = self.robot_origin

#         return int(offset_x + robot_x), int(offset_y - robot_y)


# class Visualizer:
#     BLACK: Tuple[int, int, int] = (0, 0, 0)
#     RED: Tuple[int, int, int] = (255, 0, 0)
#     WHITE: Tuple[int, int, int] = (255, 255, 255)

#     def __init__(self, world: World) -> None:
#         """
#         Note: while the Robot and World have the origin in the center of the
#         visualization, rendering places (0, 0) in the top left corner.
#         """
#         pygame.init()
#         pygame.font.init()
#         self.world = world
#         self.screen = pygame.display.set_mode((world.width, world.height))
#         pygame.display.set_caption('Gherkin Challenge')
#         self.font = pygame.font.SysFont('freesansbolf.tff', 30)

#         self.border_thickness: float = 5. # pixels
#         self.wall_thickness: float = 10. # pixels

#     def display_world(self, walls: bool = True, border: bool = True) -> None:
#         """
#         Display the world
#         """
#         goal = self.world.convert_to_display(self.world.goal) 
#         pygame.draw.circle(self.screen, self.RED, goal, 6)

#         if border == True:
#             ''' added border of thickness 5 pixels to the world
#                 can be easily configured for other thickness by changing the value of 'self.border_thickness'
#             '''

#             pygame.draw.rect(self.screen, self.BLACK, (0,0,self.world.width,self.border_thickness)) # Top border
#             pygame.draw.rect(self.screen, self.BLACK, (0,self.world.height-self.border_thickness,self.world.width,self.border_thickness)) # Bottom border
#             pygame.draw.rect(self.screen, self.BLACK, (0,0,self.border_thickness,self.world.height)) # left border
#             pygame.draw.rect(self.screen, self.BLACK, (self.world.width-self.border_thickness,0,self.border_thickness,self.world.height)) # right border 


#         if walls == True:

#             '''creating a C shaped wall around the robot'''

#             pygame.draw.rect(self.screen, self.BLACK, pygame.Rect((85,70,self.wall_thickness,160))) # center wall --> rect starting point (x,y) = (85,75) --> 10 pixels thick and 160 pixels height  
#             pygame.draw.rect(self.screen, self.BLACK, pygame.Rect((85,70,75+15,self.wall_thickness))) # top wall --> rect starting point (x,y) = (85,75) --> 10 pixels thick and 90 pixels width
#             pygame.draw.rect(self.screen, self.BLACK, pygame.Rect((85,230,75+15,self.wall_thickness))) # bottom wall --> rect starting point (x,y) = (85,230) --> 10 pixels thick and 90 pixels width


#     def display_robot(self, robot: Robot) -> None:
#         """
#         Display the robot
#         """
#         j0 = self.world.robot_origin
#         j1 = self.world.convert_to_display(robot.joint_1_pos()) 
#         j2 = self.world.convert_to_display(robot.joint_2_pos()) 
#         # Draw joint 0
#         pygame.draw.circle(self.screen, self.BLACK, j0, 4)
#         # Draw link 1
#         pygame.draw.line(self.screen, self.BLACK, j0, j1, 2)
#         # Draw joint 1
#         pygame.draw.circle(self.screen, self.BLACK, j1, 4)
#         # Draw link 2
#         pygame.draw.line(self.screen, self.BLACK, j1, j2, 2)
#         # Draw joint 2
#         pygame.draw.circle(self.screen, self.BLACK, j2, 4)

#     def update_display(self, robot: Robot, success: bool) -> bool:
#         for event in pygame.event.get():
#             # Keypress
#             if event.type == pygame.KEYDOWN:
#                 # Escape key
#                 if event.key == pygame.K_ESCAPE:
#                     return False
#             # Window Close Button Clicked
#             if event.type == pygame.QUIT:
#                 return False

#         self.screen.fill(self.WHITE)
        
#         # walls and borders can be optional
#         walls = False
#         border = True

#         self.display_world(walls, border)

#         self.display_robot(robot)

#         if success:
#             text = self.font.render('Success!', True, self.BLACK)
#             self.screen.blit(text, (1, 1))

#         pygame.display.flip()

#         return True

#     def cleanup(self) -> None:
#         pygame.quit()


# class Controller:
#     def __init__(self, goal: Tuple[int, int]) -> None:
#         self.goal = goal
#         self.goal_theta_0, self.goal_theta_1 = Robot.inverse(self.goal[0], self.goal[1])

#     def step(self, robot: Robot) -> Robot:
#         """
#         Simple P controller
#         Change the value of Kp to work with reinforcement learning 
#         """
#         theta_0_error = self.goal_theta_0 - robot.theta_0
#         theta_1_error = self.goal_theta_1 - robot.theta_1

#         Kp = 1.7

#         robot.theta_0 += theta_0_error * Kp
#         robot.theta_1 += theta_1_error * Kp

#         return robot


# class Runner:
#     def __init__(
#         self,
#         robot: Robot,
#         controller: Controller,
#         world: World,
#         vis: Visualizer
#     ) -> None:
#         self.robot = robot
#         self.controller = controller
#         self.world = world
#         self.vis = vis

#     def run(self) -> None:
#         running = True
#         count = 0

#         while count<1000:
#             # Step the controller
#             self.robot = self.controller.step(self.robot) 

#             # Check collisions
#             # TODO

#             # Check success
#             success = self.check_success(self.robot, self.world.goal)

#             # Update the display
#             running = self.vis.update_display(self.robot, success)
            
#             # sleep for Robot DT seconds, to force update rate
#             time.sleep(self.robot.DT)

#             count+=1

#     @staticmethod
#     def check_success(robot: Robot, goal: Tuple[int, int]) -> bool:
#         """
#         Check that robot's joint 2 is very close to the goal.
#         Don't not use exact comparision, to be robust to floating point calculations.
#         """
#         return np.allclose(robot.joint_2_pos(), goal, atol=0.25)

#     def cleanup(self) -> None:
#         self.vis.cleanup()

# def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[int, int]:
#     """
#     Generate a random goal that is reachable by the robot arm
#     """
#     # Ensure theta is not 0
#     theta = (np.random.random() + np.finfo(float).eps) * 2 * np.pi
#     # Ensure point is reachable
#     r = np.random.uniform(low=min_radius, high=max_radius)

#     x = int(r * np.cos(theta))
#     y = int(r * np.sin(theta))

#     return x, y


# def main_simulate() -> None:
#     height = 300
#     width = 300

#     robot_origin = (int(width / 2), int(height / 2))
#     goal = generate_random_goal(Robot.min_reachable_radius(), Robot.max_reachable_radius())

#     robot = Robot()
#     controller = Controller(goal) 
#     world = World(width, height, robot_origin, goal)
#     vis = Visualizer(world)

#     runner = Runner(robot, controller, world, vis)

#     try:
#         runner.run()
#     except AssertionError as e:
#         print(f'ERROR: {e}, Aborting.')
#     except KeyboardInterrupt:
#         pass
#     finally:
#         runner.cleanup()


# if __name__ == '__main__':
#     main()
