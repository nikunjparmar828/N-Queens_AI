import numpy as np
import scipy
from scipy.integrate import solve_ivp
from math import cos, pi, sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation


m1 = 1
m2 = 1
l1 = 0.5
l2 = 0.5
r1 = l1/2
r2 = l2/2
I1 = 0.098
I2 = 0.098
g = 9.81

class one_link:

    def __init__(self, m1, r1, I1, m2, r2, I2):
        # self.m1 = m1
        # self.r1 = r1
        # self.I1 = I1
        # self.m2 = m2
        # self.r2 = r2
        # self.I2 = I2
        
        # self.g = -9.81
        self.torq1 = 0
        self.torq2 = 0

    def update_torq(self, T):
        self.torq1 = T[0]
        self.torq2 = T[1]

    def get_next_state(self, t, y):
        theta_1 = y[0]
        theta_dot_1 = y[1]
        theta_2 = y[2]
        theta_dot_2 = y[3]
        
        theta_ddot_1 = (I2*self.torq1 - I2*self.torq2 + m2*r2**2*self.torq1 - m2*r2**2*self.torq2 + l1*m2**2*r2**3*theta_dot_1**2*sin(theta_2) + l1*m2**2*r2**3*theta_dot_2**2*sin(theta_2) + g*l1*m2**2*r2**2*sin(theta_1) + \
                        I2*g*l1*m2*sin(theta_1) + I2*g*m1*r1*sin(theta_1) - l1*m2*r2*self.torq2*cos(theta_2) + \
                        2*l1*m2**2*r2**3*theta_dot_1*theta_dot_2*sin(theta_2) + \
                        l1**2*m2**2*r2**2*theta_dot_1**2*cos(theta_2)*sin(theta_2) - g*l1*m2**2*r2**2*sin(theta_1 + \
                        theta_2)*cos(theta_2) + I2*l1*m2*r2*theta_dot_1**2*sin(theta_2) + \
                        I2*l1*m2*r2*theta_dot_2**2*sin(theta_2) + g*m1*m2*r1*r2**2*sin(theta_1) + \
                        2*I2*l1*m2*r2*theta_dot_1*theta_dot_2*sin(theta_2))/(- l1**2*m2**2*r2**2*cos(theta_2)**2 + \
                        l1**2*m2**2*r2**2 + I2*l1**2*m2 + m1*m2*r1**2*r2**2 + I1*m2*r2**2 + I2*m1*r1**2 + I1*I2)
        
        theta_ddot_2 = -(I2*self.torq1 - I1*self.torq2 - I2*self.torq2 - l1**2*m2*self.torq2 - m1*r1**2*self.torq2 + m2*r2**2*self.torq1 - m2*r2**2*self.torq2 +
                        l1*m2**2*r2**3*theta_dot_1**2*sin(theta_2) + l1**3*m2**2*r2*theta_dot_1**2*sin(theta_2) +
                        l1*m2**2*r2**3*theta_dot_2**2*sin(theta_2) - g*l1**2*m2**2*r2*sin(theta_1 + theta_2) -
                        I1*g*m2*r2*sin(theta_1 + theta_2) + g*l1*m2**2*r2**2*sin(theta_1) + I2*g*l1*m2*sin(theta_1) +
                        I2*g*m1*r1*sin(theta_1) + l1*m2*r2*self.torq1*cos(theta_2) - 2*l1*m2*r2*self.torq2*cos(theta_2) +
                        2*l1*m2**2*r2**3*theta_dot_1*theta_dot_2*sin(theta_2) +
                        2*l1**2*m2**2*r2**2*theta_dot_1**2*cos(theta_2)*sin(theta_2) +
                        l1**2*m2**2*r2**2*theta_dot_2**2*cos(theta_2)*sin(theta_2) - g*l1*m2**2*r2**2*sin(theta_1 +
                        theta_2)*cos(theta_2) + g*l1**2*m2**2*r2*cos(theta_2)*sin(theta_1) -
                        g*m1*m2*r1**2*r2*sin(theta_1 + theta_2) + I1*l1*m2*r2*theta_dot_1**2*sin(theta_2) +
                        I2*l1*m2*r2*theta_dot_1**2*sin(theta_2) + I2*l1*m2*r2*theta_dot_2**2*sin(theta_2) +
                        g*m1*m2*r1*r2**2*sin(theta_1) +
                        2*l1**2*m2**2*r2**2*theta_dot_1*theta_dot_2*cos(theta_2)*sin(theta_2) +
                        l1*m1*m2*r1**2*r2*theta_dot_1**2*sin(theta_2) +
                        2*I2*l1*m2*r2*theta_dot_1*theta_dot_2*sin(theta_2) +
                        g*l1*m1*m2*r1*r2*cos(theta_2)*sin(theta_1))/(- l1**2*m2**2*r2**2*cos(theta_2)**2 +
                        l1**2*m2**2*r2**2 + I2*l1**2*m2 + m1*m2*r1**2*r2**2 + I1*m2*r2**2 + I2*m1*r1**2 + I1*I2)


        return [theta_dot_1, theta_ddot_1,theta_dot_2, theta_ddot_2]

    def ODE45(self, tf, theta_1, theta_dot_1, theta_2, theta_dot_2):
        t_eval = [tf]
        # t_eval = np.linspace(0, 5, num=500)
        sol = solve_ivp(self.get_next_state, [0, tf], [theta_1, theta_dot_1, theta_2, theta_dot_2], t_eval= t_eval, method='RK45', dense_output=True)
        return sol

# def get_coords(th):
#     """Return the (x, y) coordinates of the bob at angle th."""
#     return 1 * np.cos(th), 1 * np.sin(th)
#
# def animate(i):
#     """Update the animation at frame i."""
#     x, y = get_coords(sol.y[0][i])
#     line.set_data([0, x], [0, y])
#     circle.set_center((x, y))


if __name__ == '__main__':
    print(cos(-2))
    sim = one_link(1, 0.5, 0.5)
    sol = sim.ODE45(5, pi/2, 0, 0,0)
    print(sol.y[0])



    #
    # L=  1
    # # Initialize the animation plot. Make the aspect ratio equal so it looks right.
    # fig = plt.figure()
    # ax = fig.add_subplot(aspect='equal')
    # theta0 = sol.y[0][0]
    # # The pendulum rod, in its initial position.
    # x0, y0 = get_coords(theta0)
    # line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
    # # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    # bob_radius = 0.08
    # circle = ax.add_patch(plt.Circle(get_coords(theta0), bob_radius,
    #                                  fc='r', zorder=3))
    # # Set the plot limits so that the pendulum has room to swing!
    # ax.set_xlim(-L * 1.2, L * 1.2)
    # ax.set_ylim(-L * 1.2, L * 1.2)
    #
    #
    #
    # print(len(sol.y[0]))
    # nframes = 500
    # interval = 10
    # ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
    #                               interval=interval)
    # # ani.save("ani")
    # ani.save('the_movie.', writer='mencoder', fps=15)
    # plt.show()
