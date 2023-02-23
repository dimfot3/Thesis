import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from numpy import random 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


class HumanMotionCreator: 
    """
    This is a class with functions to generate, plot and export
    human motion. The motion is created as smooth 2d longitudinal 
    and lateral movemnt alongside with a corresponding yaw rotation.
    """
    def __init__(self, xlims, ylims, num_p, num_interp, speed_lim, speed_std):
        """
        The constructor of HumanMotionCreator
        
        :param xlims: 2d vector that contants x limits. Ex. [xmin, xmax]
        :param ylims: 2d vector that contants y limit. Ex. [ymin, ymax]
        :param num_p: the number of distinct destinations the human will achieve
        :param num_interp: The total interpolated points for human's movement
        :param speed_lim: speed limit for human's movement
        :param speed_std: the standard variation of human's speed
        """
        self.xlims = xlims
        self.ylims = ylims
        self.num_p = num_p
        self.num_interp = num_interp
        self.speed_lim = speed_lim
        self.speed_std = speed_std

    def find_motion(self):
        """
        This function finds the motion of human.

        :return: human motion in Nx4 matrix where columns are [timestamp, x, y, yaw]
        """
        # Generate the x and y coordinates
        x = [random.uniform(self.xlims[0], self.xlims[1]) for i in range(self.num_p)]
        y = [random.uniform(self.ylims[0], self.ylims[1]) for i in range(self.num_p)]
        # this makes the motion start's = end's
        x.append(x[0])
        y.append(y[0])
        # find the timestamps that follow speed limits
        dx = np.diff(x)
        dy = np.diff(y)
        timestamps = [0] * len(x)
        for i in range(dx.shape[0]):
            curr_speed_limit = (1 - self.speed_std / 2 + np.random.rand() * self.speed_std) * self.speed_lim
            timestamps[i+1] = timestamps[i] + np.sqrt(dx[i]**2 + dy[i]**2) / curr_speed_limit
        # find the B-splines of the paths
        t = np.array(timestamps)
        spline_x = make_interp_spline(t, x, k=2)
        spline_y = make_interp_spline(t, y, k=2)
        # Generate a set of new timestamps with a finer time resolution
        num_interp_points = self.num_interp
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_interp_points)
        # Interpolate the x and y values at the new timestamps
        interp_x = spline_x(new_timestamps)
        interp_y = spline_y(new_timestamps)
        inter_yaw = np.insert(np.arctan2(np.diff(interp_y), np.diff(interp_x)), 0, 0)
        return np.stack([new_timestamps, interp_x, interp_y, inter_yaw], axis=1)

    def plot_motion(self, motion_mat):
        """
        This function finds the motion of human.

        :return: human motion in Nx4 matrix where columns are [timestamp, x, y, yaw]
        :return: None
        """
        # Define the number of frames in the animation
        num_frames = motion_mat.shape[0]
        # Create the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim([min(motion_mat[:, 1]), max(motion_mat[:, 1])])
        ax.set_ylim([min(motion_mat[:, 2]), max(motion_mat[:, 2])])
        plt.plot(motion_mat[:, 1], motion_mat[:, 2])
        vector_length = 0.8
        # Define the update function
        def update(frame, interp_x, interp_y, inter_yaw):
            x = interp_x[frame] 
            y = interp_y[frame]
            yaw = inter_yaw[frame]
            line = ax.quiver(x, y, vector_length * math.cos(yaw), vector_length * math.sin(yaw), angles='xy', scale_units='xy', scale=1)
            return line,
        # Create the animation
        ani = animation.FuncAnimation(fig, lambda frame: update(frame, motion_mat[:, 1], motion_mat[:, 2], motion_mat[:, 3]), 
        frames=range(num_frames - 1), interval=np.diff(motion_mat[:, 0]).mean()*1000, blit=True)
        # Show the animation
        plt.show()


if __name__ == '__main__':
    hm_cr = HumanMotionCreator([-10, 10], [-10, 10], 5, 2000, 2.2, 0.2)
    motion = hm_cr.find_motion()
    hm_cr.plot_motion(motion)