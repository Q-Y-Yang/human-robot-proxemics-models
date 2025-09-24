import csv
import numpy as np
import argparse
import os
import math
import pandas as pd
# import quaternion
# import matplotlib.pyplot as plt
# from pytransform3d import transformations as pt
# from pytransform3d.transform_manager import TransformManager


def load_csv(path2csv):
    rows = []

    with open(path2csv, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return rows

def load_joint_position(rows, joint, index):
    if rows[index[0]][index[1]] == joint:        
        if rows[index[0]+2][index[1]+4] == "Position" and rows[index[0]+3][index[1]+4] == "X":
            #psition = [X, Y, Z]
            a = '%f' % float(rows[index[0]+4][index[1]+2])
            # print(index[0]+4, index[1]+2, a)
            joint_position = np.array([float(rows[index[0]+4][index[1]+4]), float(rows[index[0]+4][index[1]+4+1]), float(rows[index[0]+4][index[1]+4+2])])

        return joint_position
    
def load_all_joint_position(rows, joint, index):
    all_joint_position = []
    if rows[index[0]][index[1]] == joint:        
        if rows[index[0]+2][index[1]+4] == "Position" and rows[index[0]+3][index[1]+4] == "X":
            for j in range(index[0]+4, len(rows)):
                if rows[j][index[1]+4] != '':
                    try:
                        all_joint_position.append([float(rows[j][index[1]+4]), float(rows[j][index[1]+4+1]), float(rows[j][index[1]+4+2])])
                    except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
                        print("ValueError:", [float(rows[j][index[1]+4]), float(rows[j][index[1]+4+1]), float(rows[j][index[1]+4+2])])

    return all_joint_position


def load_joint_rotation(rows, joint, index):
    if rows[index[0]][index[1]] == joint:
        if rows[index[0]+2][index[1]] == "Rotation" and rows[index[0]+3][index[1]] == "X":
            #Q = (w, x, y, z)
            joint_rotation = np.quaternion(float(rows[index[0]+4][index[1]+3]), float(rows[index[0]+4][index[1]]), float(rows[index[0]+4][index[1]+1]), float(rows[index[0]+4][index[1]+2]))

        return joint_rotation


def load_joint_rotations(rows, joint, index):
    data = []
    if rows[index[0]][index[1]] == joint and \
       rows[index[0]+2][index[1]] == "Rotation" and \
       rows[index[0]+3][index[1]] == "X":

        for j in range(index[0]+4, len(rows)):
            try:
                vals = rows[j][index[1] : index[1]+4]  # 4 values: X, Y, Z, W
                data.append([float(v) for v in vals] if all(vals) else None)
            except (ValueError, IndexError):
                data.append(None)
    # print("length of data:", len(data))
    return data
  

def compute_relative_pose(position_A, quaternion_A, position_B, quaternion_B):
    # compute relative pose (from B to A)
    relative_position = position_A - position_B

    quaternion_B_inverse = np.quaternion.conjugate(quaternion_B)

    relative_orientation = quaternion_B_inverse * quaternion_A

    return relative_position, relative_orientation

def compute_relative_position(position_A, position_B):
    # compute relative pose (from B to A)
    # print(len(position_A), len(position_B))
    if isinstance(position_A, list):
        relative_position = [np.array(position_A[i]) - np.array(position_B[i]) for i in range(len(position_A))]
    else:
        relative_position = position_A - position_B

    return relative_position

def find_csv(mode, input_folder_path):
    assert mode in ["train", "test", "all"], "Mode must be 'train' or 'test'"
    
    # Find relevant CSV files
    csv_files = [f for f in os.listdir(input_folder_path)
                 if f.endswith(".csv")]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for pattern: in {input_folder_path}")
    
    # 80/20 split
    split_idx = math.ceil(0.8 * len(csv_files))
    if mode == "train":
        selected_files = csv_files[:split_idx]  
    elif mode == "test":   
        selected_files = csv_files[split_idx:]
    else:
        selected_files = csv_files

    return selected_files

def load_joint_positions(rows, joint, index):
    positions = []
    if rows[index[0]][index[1]] == joint and \
       rows[index[0]+2][index[1]+4] == "Position" and \
       rows[index[0]+3][index[1]+4] == "X":

        for j in range(index[0]+4, len(rows)):
            try:
                vals = rows[j][index[1]+4 : index[1]+7]
                if all(vals):  # check none are empty strings
                    positions.append([float(v) for v in vals])
                else:
                    positions.append(None)
            except (ValueError, IndexError):
                positions.append(None)

    return positions

def main(args):
    path2csv = args.input_folder_path + args.group_name + "_group_csv/Group" + args.group_number + "/"
    outdir = args.output_folder_path + args.mode + '/' + args.group_name + "_group_csv/Group" + args.group_number 
    os.makedirs(outdir, exist_ok=True)

    csv_files = find_csv(mode=args.mode, input_folder_path=path2csv)
    print(csv_files)
    for file in csv_files:
        rows = load_csv(path2csv + file)
        label_row = rows[2]
        print(file)

        # Define joint names and indices in a dictionary
        if args.group_name == "human":
            joints = {
                "p1_Chest": [2, label_row.index('p1_Chest')],        #[2, 28],
                "p2_Chest": [2, label_row.index('p2_Chest')],        #[2, 286],
                "p3_Chest": [2, label_row.index('p3_Chest')],        #[2, 544],
                "p4_Chest": [2, label_row.index('p4_Chest')]         #[2, 802],
            }

        elif args.group_name == "robot":
            joints = {
                "p1_Chest": label_row.index('p1_Chest'),        #[2, 28],
                "p2_Chest": label_row.index('p2_Chest'),        #[2, 286],
                "p3_Chest": label_row.index('p3_Chest'),        #[2, 544],
                "Rigid Body 3": label_row.index('Rigid Body 3')     #[2, 776]
            }

        # Load joint positions
        # joint_positions = {
        #     name: load_all_joint_position(rows, name, idx)
        #     for name, idx in joints.items()
        # }
        # Load all joint data
        raw = {n: load_all_joint_position(rows, n, i) for n, i in joints.items()}

        # Transpose and filter valid rows
        joint_positions = {n: [] for n in joints}
        for row_vals in zip(*raw.values()):
            if all(v is not None for v in row_vals):
                for name, val in zip(joint_positions, row_vals):
                    joint_positions[name].append(val)

        # Compute all pairwise relative positions (B relative to A)

        for ref_name, ref_pos in joint_positions.items():
            rel_dict = {}
            for target_name, target_pos in joint_positions.items():
                if ref_name != target_name:
                    rel = compute_relative_position(target_pos, ref_pos)
                    print("target_name:", target_name)
                    print("ref_name:", ref_name)
                    rel_dict[target_name] = rel

            # Save each to a CSV file
            # print(rel_dict)
            filename = f"{outdir}/rel_to_{ref_name}_"+ file
            pd.DataFrame(rel_dict).to_csv(filename, index=False)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gn', '--group_name', type=str, default='human')
    parser.add_argument('-n', '--group_number', type=str, default="1")
    parser.add_argument('-o', '--output_folder_path', type=str, default="processed/")
    parser.add_argument('-i', '--input_folder_path', type=str, default="data_utils/")
    parser.add_argument('-m', '--mode', type=str, default="test")

    main(parser.parse_args())

