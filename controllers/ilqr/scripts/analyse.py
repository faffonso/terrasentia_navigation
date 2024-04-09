import rosbag
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

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

def plot_trajectory(trajectory_xs, trajectory_ys, x_ref, y_ref, heading_ref):
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_xs, trajectory_ys, label='Trajectory')
    plt.scatter(x_ref, y_ref, color='red', marker='o', label='Reference Point')  # Change marker to 'o' for a circle

    # Plot arrows for goal orientation
    for x, y, heading in zip(x_ref, y_ref, heading_ref):
        arrow_length = 0.8  # Adjust arrow length as needed
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)

        plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')

    #max_val = max(max(trajectory_xs), max(trajectory_ys), max(x_ref), max(y_ref))
    
    plt.xlim(-2.0, 2.0)
    plt.ylim(-7.0, 7.0)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_control(control_v, control_omega):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # Increased height to accommodate larger labels

    axes[0].step(range(len(control_v)), control_v, where='mid')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=2)
    axes[0].set_ylabel('$v$ [m/s]', fontsize=12)  # Adjust font size as needed
    axes[0].grid(True)

    axes[1].step(range(len(control_omega)), control_omega, where='mid')
    axes[1].axhline(0, color='k', linestyle='--', linewidth=2)
    axes[1].set_ylabel('$\omega$ [rad/s]', fontsize=12)  # Adjust font size as needed
    axes[1].grid(True)

    plt.xlabel('Step $k$', fontsize=12)  # Adjust font size as needed
    plt.xlim(0, len(control_v))

    fig.align_ylabels(axes)  # Align y-axis labels vertically

    plt.show()

def plot_ilqr_time(ilqr_time):
    plt.figure(figsize=(10, 4))
    plt.plot(ilqr_time, label='ilqr_time')
    plt.xlabel('Time Step')
    plt.ylabel('Time (s)')
    plt.title('ILQR Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bag_path = '../bags/jint/04-09-ilqr-curved-16-02.bag'
    # bag_path = '../../nmpc/bags/04-09-curved-nmpc-13-25.bag'
    bag = rosbag.Bag(bag_path)
    trajectory_xs, trajectory_ys, control_v, control_omega, ilqr_time, x_ref, y_ref, heading_ref = extract_data(bag)
    
    plot_trajectory(trajectory_xs, trajectory_ys, x_ref, y_ref, heading_ref)
    plot_control(control_v, control_omega)
    plot_ilqr_time(ilqr_time)

    tot = 0
    ind = 0
    tot2 = 0
    for t in ilqr_time:
        #tot += 1/t
        tot2 += t
        ind += 1

    print(tot2/ind)
    #print(ind/tot)

    bag.close()
