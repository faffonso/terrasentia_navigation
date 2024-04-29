import rosbag
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion, quaternion_from_euler
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
            orientation_q = msg.pose.orientation
            _, _, heading = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
            heading_ref.append(heading)

    return xs, ys, v, omega, ilqr_time, x_ref, y_ref, heading_ref

def plot_trajectory(trajectory_xs1, trajectory_ys1, x_ref1, y_ref1, heading_ref1,
                    trajectory_xs2, trajectory_ys2, x_ref2, y_ref2, heading_ref2,
                    trajectory_xs3, trajectory_ys3, x_ref3, y_ref3, heading_ref3,
                    background_image_path, opacity=1.0):
    
    plt.figure(figsize=(16, 7))

    # Load the background image
    background_image = Image.open(background_image_path)

    # Mirror the background image horizontally
    mirrored_image = background_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Calculate the aspect ratio of the image
    aspect_ratio = mirrored_image.width / mirrored_image.height



    # Display the background image
    plt.imshow(mirrored_image, extent=[-9.0, 9.0, -5.06, 5.06], aspect='auto', alpha=opacity)

    plt.scatter(-8.3, -0.32, marker='o', label='Initial point', s=100, color='red')

def plot_trajectory(trajectory_xs1, trajectory_ys1, x_ref1, y_ref1, heading_ref1,
                    trajectory_xs2, trajectory_ys2, x_ref2, y_ref2, heading_ref2,
                    trajectory_xs3, trajectory_ys3, x_ref3, y_ref3, heading_ref3,
                    background_image_path, opacity=1.0):
    
    plt.figure(figsize=(14, 4))

    # Load the background image
    background_image = Image.open(background_image_path)

    # Mirror the background image horizontally
    mirrored_image = background_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Calculate the aspect ratio of the image
    aspect_ratio = mirrored_image.width / mirrored_image.height

    # Display the background image
    plt.imshow(mirrored_image, extent=[-9.0, 9.0, -5.06, 5.06], aspect='auto', alpha=opacity)

    plt.plot(trajectory_ys1, trajectory_xs1, label='Trajectory 1', linewidth=3.5, linestyle='-', zorder=1, color='blue')
    plt.plot(trajectory_ys2, trajectory_xs2, label='Trajectory 2', linewidth=2.5, linestyle='-', zorder=2, color='orange')
    plt.plot(trajectory_ys3, trajectory_xs3, label='Trajectory 3', linewidth=1.25, linestyle='-', zorder=3, color='red')


    # Calculate the limits based on the aspect ratio
    ylim = [-2.2, 1.2]
    xlim = [-9.0 , 9.0]
    
    plt.scatter(-8.3, -0.32, marker='o', label='Initial point', s=150, color='black', zorder=4)

    plt.xlabel('$Y_g$ [m]', fontsize=14)
    plt.ylabel('$X_g$ [m]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_control(control_v1, control_omega1,
                control_v2, control_omega2,
                control_v3, control_omega3):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Increased height to accommodate larger labels

    axes[0].step(range(len(control_v1)), control_v1, where='mid')
    axes[0].step(range(len(control_v2)), control_v2, where='mid')
    axes[0].step(range(len(control_v3)), control_v3, where='mid')

    axes[0].axhline(0, color='k', linestyle='--', linewidth=2)
    axes[0].set_ylabel('$v$ [m/s]', fontsize=12)  # Adjust font size as needed
    axes[0].grid(True)

    axes[1].step(range(len(control_omega1)), control_omega1, where='mid')
    axes[1].step(range(len(control_omega2)), control_omega2, where='mid')
    axes[1].step(range(len(control_omega3)), control_omega3, where='mid')
    axes[1].axhline(0, color='k', linestyle='--', linewidth=2)
    axes[1].set_ylabel('$\omega$ [rad/s]', fontsize=12)  # Adjust font size as needed
    axes[1].grid(True)

    plt.xlabel('Step $k$', fontsize=12)  # Adjust font size as needed
    plt.xlim(0, len(control_v3))

    fig.align_ylabels(axes)  # Align y-axis labels vertically

    plt.show()

def plot_ilqr_time(ilqr_time1, ilqr_time2, ilqr_time3):
    plt.figure(figsize=(8, 5))
    plt.plot(ilqr_time1, label='ilqr_time')
    plt.plot(ilqr_time2, label='ilqr_time')
    plt.plot(ilqr_time3, label='ilqr_time')


    plt.xlabel('Time Step')
    plt.ylabel('Time (s)')
    plt.title('ILQR Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bag_path1 = '../bags/04-11-0noise-11-32.bag'
    bag_path2 = '../bags/04-11-0noise-12-01.bag'
    bag_path3 = '../bags/04-11-0noise-13-41.bag'


    bag1 = rosbag.Bag(bag_path1)
    bag2 = rosbag.Bag(bag_path2)
    bag3 = rosbag.Bag(bag_path3)

    trajectory_xs1, trajectory_ys1, control_v1, control_omega1, ilqr_time1, x_ref1, y_ref1, heading_ref1 = extract_data(bag1)
    trajectory_xs2, trajectory_ys2, control_v2, control_omega2, ilqr_time2, x_ref2, y_ref2, heading_ref2 = extract_data(bag2)
    trajectory_xs3, trajectory_ys3, control_v3, control_omega3, ilqr_time3, x_ref3, y_ref3, heading_ref3 = extract_data(bag3)

    background_image_path = 'maps/jint_noise_map_skel.jpg'

    plot_trajectory(trajectory_xs1, trajectory_ys1, x_ref1, y_ref1, heading_ref1,
                    trajectory_xs2, trajectory_ys2, x_ref2, y_ref2, heading_ref2,
                    trajectory_xs3, trajectory_ys3, x_ref3, y_ref3, heading_ref3,
                    background_image_path)
    
    
    plot_control(control_v1, control_omega1,
                 control_v2, control_omega2,
                 control_v3, control_omega3,)
    plot_ilqr_time(ilqr_time1, ilqr_time2, ilqr_time3)

    tot1 = 0
    ind1 = 0

    tot2 = 0
    ind2 = 0

    tot3 = 0
    ind3 = 0

    for t in ilqr_time1:
        tot1 += t
        ind1 += 1
    
    for t in ilqr_time2:
        tot2 += t
        ind2 += 1

    for t in ilqr_time3:
        tot3 += t
        ind3 += 1

    print(tot1/ind1)
    print(tot2/ind2)
    print(tot3/ind3)

    #print(ind/tot)

    bag1.close()
    bag2.close()
    bag3.close()
