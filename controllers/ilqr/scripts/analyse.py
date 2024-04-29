import rosbag
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def extract_data(bag):
    timestamps = []
    xs = []
    ys = []

    x_ref = []
    y_ref = []
    heading_ref =  []

    v = []
    omega = []

    ilqr_time = []
    
    for topic, msg, t in bag.read_messages():
        if topic == '/terrasentia/will/odom':
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)

        elif topic == '/terrasentia/cmd_vel':
            v.append(msg.twist.linear.x)
            omega.append(msg.twist.angular.z)

        elif topic == '/terrasentia/ilqr_time':
            ilqr_time.append(msg.time)

        elif topic == '/terrasentia/goal':
            x_ref.append(msg.pose.position.x)
            y_ref.append(msg.pose.position.y)

    return xs, ys, v, omega, ilqr_time, x_ref, y_ref

def plot_trajectory(trajectory_xs, trajectory_ys, x_ref, y_ref, heading_ref, background_image_path, opacity=1.0):
    plt.figure(figsize=(16, 4.57))

    # Load the background image
    background_image = Image.open(background_image_path)

    # Mirror the background image horizontally
    mirrored_image = background_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Display the background image
    plt.imshow(mirrored_image, extent=[-7.5, 7.5, -4.21, 4.21], aspect='auto', alpha=opacity)

    plt.plot(trajectory_ys, trajectory_xs, label='Trajectory', linewidth=2.0)
    plt.scatter(y_ref, x_ref, color='red', marker='o', label='Waypoints', s=25)

    plt.xlabel('$Y_g$ [m]', fontsize=17)
    plt.ylabel('$X_g$ [m]', fontsize=17)
    plt.xlim(-7.5, 7.5)
    plt.ylim(-2.0, 1.5)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()

def plot_controls(control_v1, control_omega1, ilqr_time1, mean1,
                  control_v2, control_omega2, ilqr_time2, mean2):
    fig, axes = plt.subplots(2, 2, figsize=(13, 4), sharex='col')

    axes[0, 0].step(range(len(control_v1)), control_v1, c='C0', linewidth=1.25)
    axes[0, 0].axhline(0, color='k', linestyle='-.', linewidth=1.75)
    axes[0, 0].set_ylabel('$v$ [m/s]', fontsize=14)
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(-0.05, 0.85)
    axes[0, 0].set_xlim(0, len(control_v1))
    axes[0, 0].set_title('iLQR', fontsize=14)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].tick_params(axis='x', labelsize=12)

    axes[0, 1].step(range(len(control_v2)), control_v2, c='C0', linewidth=1.25)
    axes[0, 1].axhline(0, color='k', linestyle='-.', linewidth=1.75)
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(-0.05, 0.85)
    axes[0, 1].set_xlim(0, len(control_v2))
    axes[0, 1].set_title('IPOPT', fontsize=14) 
    axes[0, 1].tick_params(axis='y', labelsize=12)
    axes[0, 1].tick_params(axis='x', labelsize=12)

    axes[1, 0].step(range(len(control_omega1)), control_omega1, color='orange', linewidth=1.25)
    axes[1, 0].axhline(0, color='k', linestyle='-.', linewidth=1.75)
    axes[1, 0].set_ylabel('$\omega$ [rad/s]', fontsize=14)
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(-0.75, 0.75)
    axes[1, 0].set_xlim(0, len(control_omega1))
    axes[1, 0].set_xlabel('Time Step', fontsize=14)
    axes[1, 0].tick_params(axis='y', labelsize=12)
    axes[1, 0].tick_params(axis='x', labelsize=12)

    axes[1, 1].step(range(len(control_omega2)), control_omega2, color='orange', linewidth=1.25)
    axes[1, 1].axhline(0, color='k', linestyle='-.', linewidth=1.75)
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(-0.75, 0.75)
    axes[1, 1].set_xlim(0, len(control_omega2))
    axes[1, 1].set_xlabel('Time Step', fontsize=14)
    axes[1, 1].tick_params(axis='y', labelsize=12)
    axes[1, 1].tick_params(axis='x', labelsize=12)

    # Hide yticks for subplots in the second row
    axes[0, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[1, 1].tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.suptitle('Control Signals', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(ilqr_time1, label='iLQR', c='C0', zorder=1, linewidth=1.25)
    plt.plot(ilqr_time2, label='IPOPT', c='orange', zorder=1, linewidth=1.25)
    plt.axhline(mean1, linestyle='-.', linewidth=1.75, label="Average Solver Time", zorder=3, c='k')
    plt.axhline(mean2, linestyle='-.', linewidth=1.75, zorder=3, c='k')
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Solver Time [s]', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    plt.xlim(0, max(len(ilqr_time1), len(ilqr_time2))-1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    bag_path1 = '../bags/04-09-ilqr-curved-16-02.bag'
    bag_path2 = '../bags/04-09-nmpc-curved-16-10.bag'
    bag1 = rosbag.Bag(bag_path1)
    bag2 = rosbag.Bag(bag_path2)

    trajectory_xs1, trajectory_ys1, control_v1, control_omega1, ilqr_time1, x_ref1, y_ref1 = extract_data(bag1)
    trajectory_xs2, trajectory_ys2, control_v2, control_omega2, ilqr_time2, x_ref2, y_ref2 = extract_data(bag2)
    background_image_path = 'maps/jint_noise_map.jpg'

    # plot_trajectory(trajectory_xs1, trajectory_ys1, x_ref1, y_ref1, [], background_image_path)
    
    plot_controls(control_v1, control_omega1, ilqr_time1, np.mean(ilqr_time1),
                  control_v2, control_omega2, ilqr_time2, np.mean(ilqr_time2))

    print(f'Av. solve time: iLQR {np.mean(ilqr_time1)}, IPOPt {np.mean(ilqr_time2)}')
    print(f'Av. solve time: iLQR {(len(ilqr_time1)-1) * 0.1}, IPOPt {(len(ilqr_time2)-1) * 0.1}')

    bag1.close()
    bag2.close()
