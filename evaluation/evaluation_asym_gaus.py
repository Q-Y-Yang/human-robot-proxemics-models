import numpy as np
from data_utils.data_extract import find_csv, load_csv
import argparse
import math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from mvem.stats import multivariate_skewnorm as mvsn
import matplotlib.pyplot as plt
from evaluation_avoidance import extract_columns, load_split_data
import matplotlib.cm as cm



def asymmetric_gaussian(x, y, xc, yc, theta,
                        sigma_h, sigma_s, sigma_r):
    """
    Compute asymmetric Gaussian at (x,y) centered on (xc, yc),
    rotated by theta, with variances sigma_h (upper half),
    sigma_r (lower half), and sigma_s (sideband).

    Parameters
    ----------
    x, y : array-like or float
        Coordinates where to evaluate the Gaussian.
    xc, yc : float
        Center of the Gaussian.
    theta : float
        Rotation angle in radians (counter-clockwise).
    sigma_h : float
        Variance used for alpha > 0.
    sigma_s : float
        Variance for the secondary axis.
    sigma_r : float
        Variance used for alpha <= 0.

    Returns
    -------
    G : array-like or float
        Value of the asymmetric Gaussian at (x,y).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Shift to center
    dx = x - xc
    dy = y - yc
    
    # Compute and normalize alpha
    alpha = np.arctan2(dy, dx) - theta + np.pi/2
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi

    # Select variance on the “upper” vs “lower” side
    sigma = np.where(alpha <= 0, sigma_r, sigma_h)

    # Precompute cos/sin
    cth = np.cos(theta)
    sth = np.sin(theta)

    # Quadratic form coefficients
    a = (cth**2)/(2*sigma**2) + (sth**2)/(2*sigma_s**2)
    b =  sth*cth/(2*sigma**2) - sth*cth/(2*sigma_s**2)
    c = (sth**2)/(2*sigma**2) + (cth**2)/(2*sigma_s**2)

    # Evaluate the Gaussian
    exponent = -( a*dx**2 + 2*b*dx*dy + c*dy**2 )
    return np.exp(exponent)



def quaternion_to_theta(qs):
    """
    Convert an array of unit quaternions to their rotation angles.
    
    Parameters
    ----------
    qs : array-like of shape (N,4)
        Quaternions in (w, x, y, z) format (one per row).
    
    Returns
    -------
    thetas : ndarray of shape (N,)
        Rotation angles in radians, in [0, π].
    """
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[0] != 4:
        raise ValueError("Input must be shape (N,4)")
    w = qs[-1, :]

        # Clamp to [-1,1] to avoid numerical round-off
    w = np.clip(w, -1.0, 1.0)
    thetas = 2.0 * np.arccos(w)
    return thetas

# Example
# q = [0.8660254, 0.0, 0.5, 0.0]  # a 60° tilt about the y-axis
# theta = quaternion_to_angle(q)
# print(f"The rotation angle is {np.degrees(theta):.2f}°")


def main(args):
    path2csv = args.input_folder_path + args.group_number
    csv_files = find_csv(mode=args.mode, input_folder_path=path2csv)
    # print(csv_files)

    sigma_h = 2
    sigma_s =4/3
    sigma_r = 1

    x = np.linspace(-2, 2, 300)
    y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(x, y)

    # Parameters
    xc, yc = 0, 0         # Center
    theta=np.pi/2
    # Compute asymmetric Gaussian values
    Z = asymmetric_gaussian(X, Y, xc, yc, theta, sigma_h, sigma_s, sigma_r)

    # Plot contour map
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, colors='black')
    plt.clabel(contour, inline=True, fontsize=8)

    plt.title('Asymmetric Gaussian Contour Map')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    mean_costs = []
    std_costs = []
    all_cost_curves = []
    for file in csv_files:
        print(f"Plotting {file}")
        rows = load_csv(path2csv + file)
        split_idx = math.ceil(0.75 * len(rows)) 
        p1_pos, p1_quat, p2_pos, p2_quat, p3_pos, p3_quat, robot_pos = load_split_data(rows, split_idx)
        print(np.array(p1_quat).shape)
        p1_theta = quaternion_to_theta(p1_quat)
        p2_theta = quaternion_to_theta(p2_quat)
        p3_theta = quaternion_to_theta(p3_quat)

        robot_pos = np.asarray(robot_pos, dtype=float) 
        # unpack robot x,y
        x = robot_pos[0, :]
        y = robot_pos[1, :]


        # pack positions and thetas
        parts = [
            (np.asarray(p1_pos, dtype=float), p1_theta+np.pi/2),
            (np.asarray(p2_pos, dtype=float), p2_theta+np.pi/2),
            (np.asarray(p3_pos, dtype=float), p3_theta+np.pi/2),
        ]

        cost_per_person = []
        for p, theta in parts:
            cost = []
            for i in range(x.shape[0]):
                c = asymmetric_gaussian(
                    x[i], y[i],
                    p[0, i], p[1, i],
                    theta[i],
                    sigma_h, sigma_s, sigma_r
                )
                cost.append(c)
            cost = np.array(cost)  # shape (N,)
            cost_per_person.append(cost)

        for i, cost in enumerate(cost_per_person):
            print(f"Cost {i} shape: {cost.shape}")

        pdfs = np.array(cost_per_person) 
        mean_cost = np.mean(pdfs, axis=1) 
        print(len(mean_cost))
        pdfs_std = np.std(pdfs, axis=1)   
        # cost_avg = np.mean(costs)
 
        mean_costs.append(mean_cost)
        std_costs.append(pdfs_std)
        all_cost_curves.append(cost_per_person)
    print(len(mean_costs))

    mean_costs = np.array(mean_costs)   # shape (12, 3)
    std_costs  = np.array(std_costs)    # shape (12, 3)

    plt.figure(figsize=(12, 6))

    num_rounds = len(all_cost_curves)
    # Sample N colors from tab20b evenly
    colors = cm.get_cmap('tab10', num_rounds)

    for round_idx, cost_set in enumerate(all_cost_curves):
        color = colors(round_idx)
        for i, cost_curve in enumerate(cost_set):
            label = f'Trial {round_idx + 1}' if i == 0 else None  # Label only first curve per round
            plt.plot(cost_curve, color=color, alpha=0.5, label=label)

    plt.title('Costs per Person (Grouped and Labeled by Round)')
    plt.xlabel('Time Step')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend(title='Trials')
    plt.tight_layout()
    # Save to PNG
    plt.savefig("baseline_curves.png", dpi=300)
    plt.show()


    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    
    persons = ['Person 1', 'Person 2', 'Person 3']
    colors = ['blue', 'green', 'black']
    markers = ['o', 's', 'D']
    X_range = np.arange(1, len(mean_costs)+1)
    
    for i in range(3):
        axs[i].errorbar(X_range, mean_costs[:, i], yerr=std_costs[:, i],
                        linestyle='none', marker=markers[i], color=colors[i],
                        ecolor=colors[i], elinewidth=2, capsize=5,
                        markerfacecolor='white', markeredgewidth=1, markersize=5,
                        label=persons[i])
        
        axs[i].set_ylabel('Average Cost', fontsize=10)
        axs[i].set_ylim(-0.05, 0.4) 
        axs[i].set_xticks(X_range)
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.5)


    axs[-1].set_xlabel('Sequence Index', fontsize=12)

    # title
    fig.suptitle('Mean ± Std Deviation of Costs for Each Person in the Group', fontsize=14)

    plt.tight_layout()  
    # Save to PNG
    plt.savefig("meanstd.png", dpi=300)
    plt.show()
        # plt.figure()
        # plt.plot(Z1, label='Part 1')
        # plt.plot(Z2, label='Part 2')
        # plt.plot(Z3, label='Part 3')
        # plt.xlabel('Sample index')
        # plt.ylabel('Asymmetric Gaussian value')
        # plt.title('Asymmetric Gaussian values at robot positions')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-gn', '--group_name', type=str, default='human')
    parser.add_argument('-n', '--group_number', type=str, default="Group8/")
    # parser.add_argument('-o', '--output_folder_path', type=str, default="processed/")
    parser.add_argument('-i', '--input_folder_path', type=str, default="processed/all/robot_group_csv/")
    parser.add_argument('-m', '--mode', type=str, default="all")

    main(parser.parse_args())