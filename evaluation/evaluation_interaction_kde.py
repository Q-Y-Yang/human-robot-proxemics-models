import joblib
from data_utils.data_extract import find_csv, load_csv
from evaluation_avoidance import load_split_data_3d
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns


def minmax_kde_density(kde):
    X = np.array(kde.tree_.data)
    # Create a grid over the data range for evaluation
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
    grid_samples = np.column_stack([xx.ravel(), yy.ravel()])
    # Evaluate density
    log_density = kde.score_samples(grid_samples)
    densities = np.exp(log_density).reshape(xx.shape)

    # Normalize density to [0, 1] over the full domain
    density_min = densities.min()
    density_max = densities.max()
    # normalized_density = (density - density_min) / (density_max - density_min)

    return density_min, density_max

def normalize_kde_density(density_min, density_max, density):
    normalized_density = (density - density_min) / (density_max - density_min)

    return normalized_density

def apply_kde_model(kde, pos):
    # Evaluate density
    log_density = kde.score_samples(pos)
    density = np.exp(log_density)
    # normalized_density = normalize_kde_density(kde, density)

    return density

# Helper to extract data
def extract_columns(dataframe, indices, split_idx):
    return [dataframe.iloc[split_idx:, idx].to_numpy() for idx in indices]

def load_robot_split_data(rows, split_idx):
    # Extract 2D position and quaternion from p1
    df = pd.DataFrame(rows)
    # split_idx = math.ceil(0.8 * df.shape[0])    #number of rows
    persons_index = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23]

    robot_positions = extract_columns(df, persons_index[18:20], split_idx)
    
    return robot_positions

def transform_to_local_frame(t1, q1, t2):
    """
    Express the position t2 in the local frame of (t1, q1)

    Parameters:
    t1: numpy array of shape (3,) - position of P1 in world frame
    q1: numpy array of shape (4,) - quaternion of P1 [x, y, z, w]
    t2: numpy array of shape (3,) - position of P2 in world frame

    Returns:
    t_rel: numpy array of shape (3,) - position of P2 in P1's local frame
    """
    r1 = R.from_quat(q1)         # Rotation from P1's local frame to world
    delta = t2 - t1              # Vector from P1 to P2 in world frame
    t_rel = r1.inv().apply(delta)  # Rotate into P1's local frame
    return t_rel


def main(args):
    path2csv = args.input_folder_path + args.group_number
    csv_files = find_csv(mode=args.mode, input_folder_path=path2csv)

    kde = joblib.load("kde_model.pkl")
    print(kde)
    density_min, density_max = minmax_kde_density(kde)

    raw_data = []
    data = []
    for file in csv_files:
        rows = load_csv(path2csv + file)
        split_idx = math.ceil(0.99 * len(rows)) 
        p1_pos, p1_quat, p2_pos, p2_quat, p3_pos, p3_quat, robot_pos, _ = load_split_data_3d(rows, split_idx, first=False)

        # Convert to float arrays if they are string arrays
        p1_pos = np.array(p1_pos, dtype=np.float32)
        p2_pos = np.array(p2_pos, dtype=np.float32)
        p3_pos = np.array(p3_pos, dtype=np.float32)
        robot_pos = np.array(robot_pos, dtype=np.float32)
        # robot_quat = np.array(robot_quat, dtype=np.float32)
        persons_pos = np.array([p1_pos, p2_pos, p3_pos])
        persons_quat = np.array([p1_quat, p2_quat, p3_quat])

        relative_positions = []

        for i in range(len(persons_pos)):
            rel = transform_to_local_frame( 
                                        np.array(persons_pos[i], dtype=np.float32), 
                                        np.array(persons_quat[i], dtype=np.float32),
                                        robot_pos)
            relative_positions.append(rel)
        relative_positions = np.array(relative_positions)

        relative_positions_2d = relative_positions[:, [0, 2]]

        densities = [apply_kde_model(kde, pos.reshape(1,-1)) for pos in relative_positions_2d]
        # print(densities)
        normalized_densities = [normalize_kde_density(density_min, density_max, densities[i]) for i in range(len(densities))]
        print(normalized_densities)      
        raw_data.append(normalized_densities)
        
    for i, row in enumerate(raw_data):
        for person_id, val in enumerate(row, 1):
            data.append({"Sample": i+1, "Interactive Person": f"{person_id}", "Density": float(val)})
    df = pd.DataFrame(data)

    # Step 3: Create the violin plot with visible strip dots
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="Interactive Person", y="Density", inner="box", hue="Interactive Person",  palette="Pastel1", legend=False)

    # Add visible red dots for each density value
    sns.stripplot(
            data=df,
            x="Interactive Person",
            y="Density",
            color="red",
            size=2,
            jitter=0.3,
            alpha=0.8,
            marker='o'
        )
    
    # Calculate mean and variance per Interactive Person
    stats = df.groupby("Interactive Person")["Density"].agg(["mean", "var"]).reset_index()

    # Annotate mean and variance on the plot
    for idx, row in stats.iterrows():
        person = row["Interactive Person"]
        mean = row["mean"]
        var = row["var"]
        
        # x-coordinate: position of the category (depends on sorting order)
        x_pos = df["Interactive Person"].sort_values().unique().tolist().index(person)
        y_pos = df[df["Interactive Person"] == person]["Density"].max() + 0.02  # slightly above max for visibility
        
        plt.text(x_pos, y_pos, f"μ={mean:.2f}\nσ²={var:.2f}", 
                ha="center", va="bottom", fontsize=8, color="black")


    plt.title("Density of Interaction Positions")
    plt.ylabel("Density")
    plt.tight_layout()
    # Save to PNG
    plt.savefig("evaluate_kde.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--group_number', type=str, default="Group8/")
    # parser.add_argument('-o', '--output_folder_path', type=str, default="processed/")
    parser.add_argument('-i', '--input_folder_path', type=str, default="processed/all/robot_group_csv/")
    parser.add_argument('-m', '--mode', type=str, default="all")

    main(parser.parse_args())