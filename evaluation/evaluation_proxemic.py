from data_utils.data_extract import find_csv, load_csv
import argparse
import math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from mvem.stats import multivariate_skewnorm as mvsn
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# Helper to extract data
def extract_columns(dataframe, indices, split_idx, first):
    if first:
        return [dataframe.iloc[:split_idx, idx].to_numpy() for idx in indices]
    else:
        return [dataframe.iloc[-1, idx] for idx in indices]

def load_split_data(rows, split_idx, first=True):
    # Extract 2D position and quaternion from p1
    df = pd.DataFrame(rows)
    # split_idx = math.ceil(0.8 * df.shape[0])    #number of rows
    persons_index = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23]

    p1_position = extract_columns(df, persons_index[0:2], split_idx, first)
    p1_quaternion = extract_columns(df, persons_index[2:6], split_idx, first)
    p2_position = extract_columns(df, persons_index[6:8], split_idx, first)
    p2_quaternion = extract_columns(df, persons_index[8:12], split_idx, first)
    p3_position = extract_columns(df, persons_index[12:14], split_idx, first)
    p3_quaternion = extract_columns(df, persons_index[14:18], split_idx, first)
    robot_position = extract_columns(df, persons_index[18:20], split_idx, first)

    return p1_position, p1_quaternion, p2_position, p2_quaternion, p3_position, p3_quaternion, robot_position

def load_split_data_3d(rows, split_idx, first=True):
    df = pd.DataFrame(rows)

    p1_position = extract_columns(df, [0, 1, 2], split_idx, first)
    p1_quaternion = extract_columns(df, [3, 4, 5, 6], split_idx, first)
    p2_position = extract_columns(df, [7, 8, 9], split_idx, first)
    p2_quaternion = extract_columns(df, [10, 11, 12, 13], split_idx, first)
    p3_position = extract_columns(df, [14, 15, 16], split_idx, first)
    p3_quaternion = extract_columns(df, [17, 18, 19, 20], split_idx, first)
    robot_position = extract_columns(df, [21, 22, 23], split_idx, first)
    robot_quaternion = extract_columns(df, [24, 25, 26, 27], split_idx, first)

    return p1_position, p1_quaternion, p2_position, p2_quaternion, p3_position, p3_quaternion, robot_position, robot_quaternion

# Function to transform parameters
def transform_params(position, quaternion, mu, shape, lmbda):

    position = position.astype(float)

    # Convert quaternion to 3D rotation matrix and extract 2D X-Z submatrix
    rotation_matrix_3d = R.from_quat(quaternion).as_matrix()    # (822, 3, 3)
    rotation_matrix_2d = rotation_matrix_3d[:, [0, 2], :][:, :, [0, 2]]     # (822, 2, 2)

    # Define standard 2D orientation (pointing up in Y direction)
    std_orientation = np.array([0, 1])
    # print(position[0])
    # Apply rotation and translation
    # rotated_orientation = rotation_matrix_2d @ std_orientation
    transformed_mu = rotation_matrix_2d @ mu + position     # (822, 2, 2)
    # transformed_shape = rotation_matrix_2d @ shape @ rotation_matrix_2d.T
    transformed_lmbda = rotation_matrix_2d @ lmbda      # (822, 2, 2)

    # Broadcast shape and rotate
    shape_batched = shape[None, :, :]                                     # (1, 2, 2)
    rot_T = np.transpose(rotation_matrix_2d, axes=(0, 2, 1))              # (822, 2, 2)
    transformed_shape = rotation_matrix_2d @ shape_batched @ rot_T       # (822, 2, 2)
    # print(np.shape(transformed_shape))
    return transformed_mu, transformed_shape, transformed_lmbda

def apply_avoidance_model(x, transformed_mu, transformed_shape, transformed_lmbda):

    pdf = mvsn.pdf(x, transformed_mu, transformed_shape, transformed_lmbda)

    return pdf

def stack_columns(cols):
    return np.stack(cols, axis=1)


def main(args):
    path2csv = args.input_folder_path + args.group_number
    csv_files = find_csv(mode=args.mode, input_folder_path=path2csv)
    # print(csv_files)

    # Parameters of the learned proxemic model from proxemic_model.ipynb
    mu = np.array([ 0.01431807, -0.30797238])
    shape = np.array( [[ 0.07838676, -0.00946168],
    [-0.00946168,  0.30178896]])
    lmbda = np.array( [0.02099886, 2.10735874])

    # mu = np.array([ 0.0170915, -0.3079768])
    # shape = np.array([[ 0.0791303,  -0.01143162],
    # [-0.01143162,  0.29992035]])
    # lmbda = np.array([-0.01057365,  2.15069084])

    # mu = np.array([ 0.01581984, -0.31069371])
    # shape = np.array([[ 0.08282744, -0.00976044], [-0.00976044,  0.30564506]])
    # lmbda = np.array([1.85214741e-03, 2.00742414e+00])

    mean_costs = []
    std_costs = []
    all_cost_curves = []

    for file in csv_files:
        print(f"Plotting {file}")
        rows = load_csv(path2csv + file)
        split_idx = math.ceil(0.75 * len(rows)) 
        p1_pos, p1_quat, p2_pos, p2_quat, p3_pos, p3_quat, robot_pos = load_split_data(rows, split_idx)

        # Group data for batch processing
        positions = [np.transpose(p1_pos), np.transpose(p2_pos), np.transpose(p3_pos)]
        quaternions = [np.transpose(p1_quat), np.transpose(p2_quat), np.transpose(p3_quat)]
        robot_pos = np.transpose(robot_pos)
        # positions = [stack_columns(p) for p in [p1_pos, p2_pos, p3_pos]]
        # quaternions = [stack_columns(q).T for q in [p1_quat, p2_quat, p3_quat]]  # shape (4, N) → .T = (N, 4)


        # Apply transformation
        results = [transform_params(pos, quat, mu, shape, lmbda)
                for pos, quat in zip(positions, quaternions)]
        transformed_mu, transformed_shape, transformed_lmbda = zip(*results)

        transformed_mu = np.array(transformed_mu)         # (3, 822, 2)
        transformed_shape = np.array(transformed_shape)   # (3, 822, 2, 2)
        transformed_lmbda = np.array(transformed_lmbda)

        cost_per_person = []
        for mu_seq, shape_seq, lmbda_seq in zip(transformed_mu, transformed_shape, transformed_lmbda):
            pdf_seq = np.array([
                mvsn.pdf(robot_pos[i], mu_seq[i], np.array(shape_seq[i]), lmbda_seq[i])
                for i in range(len(robot_pos))
            ])
            
            cost_per_person.append(pdf_seq)  # shape (3, 822)

        pdfs = np.array(cost_per_person)  # shape (3, 822)
        mean_cost = np.mean(pdfs, axis=1)  # shape (3,) mean over time
        pdfs_std = np.std(pdfs, axis=1)    # shape: (3, )
        # cost_avg = np.mean(costs)
        print(pdfs_std)
        mean_costs.append(mean_cost)
        std_costs.append(pdfs_std)
        all_cost_curves.append(cost_per_person)  # cost_per_person is a list of 3 arrays, each shape ≈ (822,)


    plt.figure(figsize=(12, 6))

    num_rounds = len(all_cost_curves)
    # Sample N colors from tab20b evenly
    colors = cm.get_cmap('tab10', num_rounds)

    for round_idx, cost_set in enumerate(all_cost_curves):
        color = colors(round_idx)
        for i, cost_curve in enumerate(cost_set):
            label = f'Trial {round_idx + 1}' if i == 0 else None  # Label only first curve per round
            plt.plot(cost_curve, color=color, alpha=0.5, label=label)

    plt.title('Costs per Person (Grouped and Labeled by Trial)')
    plt.xlabel('Time Step')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend(title='Trials')
    plt.tight_layout()
    # Save to PNG
    plt.savefig("curves.png", dpi=300)
    plt.show()



    print(len(mean_costs))
    X_range = np.arange(1, len(mean_costs)+1)

    mean_costs = np.array(mean_costs)
    std_costs = np.array(std_costs)
    print(mean_costs)
    print(mean_costs[:,2])

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    persons = ['Person 1', 'Person 2', 'Person 3']
    colors = ['blue', 'green', 'black']
    markers = ['o', 's', 'D']

    # subplots
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

    fig.suptitle('Mean ± Std Deviation of Costs for Each Person in the Group', fontsize=14)

    plt.tight_layout()  
    # Save to PNG
    plt.savefig("meanstd.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-gn', '--group_name', type=str, default='human')
    parser.add_argument('-n', '--group_number', type=str, default="Group8/")
    # parser.add_argument('-o', '--output_folder_path', type=str, default="processed/")
    parser.add_argument('-i', '--input_folder_path', type=str, default="processed/all/robot_group_csv/")
    parser.add_argument('-m', '--mode', type=str, default="all")

    main(parser.parse_args())