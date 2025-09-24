import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
from scipy.spatial.transform import Rotation as R

def read_csv(path2csv):
    '''
    Read data from a csv file
    '''
    rows = []
    if not os.path.exists(path2csv):
        print("Error: The file does not exist.")
        return rows
    
    with open(path2csv, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)
    
    return rows

def save_csv(out_path, filename, data):
    '''
    Save data to a csv file
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_path + filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

    print("Data saved to:", out_path + filename)

def calc_relative_position(subject_pos, other_pos):
    '''
    Calculate the relative position of the other person to the subject
    subject_pos: [x, y, z]
    other_pos: [x, y, z]   
    '''
    return [other_pos[0] - subject_pos[0], other_pos[1] - subject_pos[1], other_pos[2] - subject_pos[2]]
                                                                                                                                                                                                                                            

def normalize_data(rows, subject):
    '''
    Normalize the data
    '''
    # indices
    # 0: p1 chest pos x
    # 1: p1 chest pos y
    # 2: p1 chest pos z
    # 3: p1 chest rot x
    # 4: p1 chest rot y
    # 5: p1 chest rot z
    # 6: p1 chest rot w
    # ... same for p2, p3, p4
    # rotation is stored in x, y, z, w - quaternion format
    # as position, we only consider x and z

    if subject not in ['p1', 'p2', 'p3', 'p4']:
        print("Error: The subject is not valid.")
        return
    
    normalized_data = []
    # %0 => x_pos, %2 => z_pos. 7 indices for each person
    for row in rows:
        # [x, y, z]
        p1_chest_pos = [float(row[0]), float(row[1]), float(row[2])]
        p1_chest_rot = [float(row[3]), float(row[4]), float(row[5]), float(row[6])]
        p2_chest_pos = [float(row[0+7]), float(row[1+7]), float(row[2+7])]
        p2_chest_rot = [float(row[3+7]), float(row[4+7]), float(row[5+7]), float(row[6+7])]
        p3_chest_pos = [float(row[0+7*2]), float(row[1+7*2]), float(row[2+7*2])]  
        p3_chest_rot = [float(row[3+7*2]), float(row[4+7*2]), float(row[5+7*2]), float(row[6+7*2])]  
        p4_chest_pos = [float(row[0+7*3]), float(row[1+7*3]), float(row[2+7*3])]
        p4_chest_rot = [float(row[3+7*3]), float(row[4+7*3]), float(row[5+7*3]), float(row[6+7*3])]
        
        # convert translation and rotation to homogeneous transformation matrix
        p1_transformation = np.eye(4)
        p1_transformation[:3, 3] = p1_chest_pos
        p1_transformation[:3, :3] = R.from_quat(p1_chest_rot).as_matrix()

        p2_transformation = np.eye(4)
        p2_transformation[:3, 3] = p2_chest_pos
        p2_transformation[:3, :3] = R.from_quat(p2_chest_rot).as_matrix()

        p3_transformation = np.eye(4)
        p3_transformation[:3, 3] = p3_chest_pos
        p3_transformation[:3, :3] = R.from_quat(p3_chest_rot).as_matrix()

        p4_transformation = np.eye(4)
        p4_transformation[:3, 3] = p4_chest_pos
        p4_transformation[:3, :3] = R.from_quat(p4_chest_rot).as_matrix()

        # check who is the subject
        buf = 0
        if subject == 'p1':
            buf = 0
        elif subject == 'p2':
            buf = 7
        elif subject == 'p3':
            buf = 14
        else:
            buf = 21

        # get subject chest pos
        subject_chest_pos = [float(row[buf]), float(row[buf+1]), float(row[buf+2])]
        # %3 => x_rot, %4 => y_rot, %5 => z_rot, %6 => w_rot
        subject_quaternion = [float(row[buf+3]), float(row[buf+4]), float(row[buf+5]), float(row[buf+6])]

        # create homogeneous transformation matrix for the subject
        subject_transformation = np.eye(4)
        subject_transformation[:3, 3] = subject_chest_pos
        subject_transformation[:3, :3] = R.from_quat(subject_quaternion).as_matrix()
       
        # calculate subject inverse transformation
        subject_inv_transformation = np.linalg.inv(subject_transformation)

        zero_pos = np.array([0, 0, 0, 1])
        # unit vector in z direction - after testing all unit vectors, this one gave the best results
        unit_vector = np.array([0, 0, 1, 0])
        # calculate the relative position of the subject to the other people
        if subject == 'p1':
            # calculate transformation from subject (p1) to p2
            p1_to_p2_transformation = np.matmul(subject_inv_transformation, p2_transformation)
            # get relative position of p2 to p1
            p2_rel_pos = np.matmul(p1_to_p2_transformation, zero_pos)[:3]
            # get relative rotation of p2 to p1
            p2_rel_rot = np.matmul(p1_to_p2_transformation, unit_vector)[:3]

            # calculate transformation from subject (p1) to p3
            p1_to_p3_transformation = np.matmul(subject_inv_transformation, p3_transformation)
            # get relative position of p3 to p1
            p3_rel_pos = np.matmul(p1_to_p3_transformation, zero_pos)[:3]
            # get relative rotation of p3 to p1
            p3_rel_rot = np.matmul(p1_to_p3_transformation, unit_vector)[:3]

            # calculate transformation from subject (p1) to p4
            p1_to_p4_transformation = np.matmul(subject_inv_transformation, p4_transformation)
            # get relative position of p4 to p1
            p4_rel_pos = np.matmul(p1_to_p4_transformation, zero_pos)[:3]
            # get relative rotation of p4 to p1
            p4_rel_rot = np.matmul(p1_to_p4_transformation, unit_vector)[:3]

            normalized_data.append([p2_rel_pos[0], p2_rel_pos[1], p2_rel_pos[2], p2_rel_rot[0], p2_rel_rot[1], p2_rel_rot[2],
                                    p3_rel_pos[0], p3_rel_pos[1], p3_rel_pos[2], p3_rel_rot[0], p3_rel_rot[1], p3_rel_rot[2],
                                    p4_rel_pos[0], p4_rel_pos[1], p4_rel_pos[2], p4_rel_rot[0], p4_rel_rot[1], p4_rel_rot[2] ])

        elif subject == 'p2':
            # calculate transformation from subject (p2) to p1
            p2_to_p1_transformation = np.matmul(subject_inv_transformation, p1_transformation)
            # get relative position of p1 to p2
            p1_rel_pos = np.matmul(p2_to_p1_transformation, zero_pos)[:3]
            # get relative rotation of p1 to p2
            p1_rel_rot = np.matmul(p2_to_p1_transformation, unit_vector)[:3]

            # calculate transformation from subject (p2) to p3
            p2_to_p3_transformation = np.matmul(subject_inv_transformation, p3_transformation)
            # get relative position of p3 to p2
            p3_rel_pos = np.matmul(p2_to_p3_transformation, zero_pos)[:3]
            # get relative rotation of p3 to p2
            p3_rel_rot = np.matmul(p2_to_p3_transformation, unit_vector)[:3]

            # calculate transformation from subject (p2) to p4
            p2_to_p4_transformation = np.matmul(subject_inv_transformation, p4_transformation)
            # get relative position of p4 to p2
            p4_rel_pos = np.matmul(p2_to_p4_transformation, zero_pos)[:3]
            # get relative rotation of p4 to p2
            p4_rel_rot = np.matmul(p2_to_p4_transformation, unit_vector)[:3]

            normalized_data.append([p1_rel_pos[0], p1_rel_pos[1], p1_rel_pos[2], p1_rel_rot[0], p1_rel_rot[1], p1_rel_rot[2],
                                    p3_rel_pos[0], p3_rel_pos[1], p3_rel_pos[2], p3_rel_rot[0], p3_rel_rot[1], p3_rel_rot[2],
                                    p4_rel_pos[0], p4_rel_pos[1], p4_rel_pos[2], p4_rel_rot[0], p4_rel_rot[1], p4_rel_rot[2] ])
        elif subject == 'p3':
            # calculate transformation from subject (p3) to p1
            p3_to_p1_transformation = np.matmul(subject_inv_transformation, p1_transformation)
            # get relative position of p1 to p3
            p1_rel_pos = np.matmul(p3_to_p1_transformation, zero_pos)[:3]
            # get relative rotation of p1 to p3
            p1_rel_rot = np.matmul(p3_to_p1_transformation, unit_vector)[:3]

            # calculate transformation from subject (p3) to p2
            p3_to_p2_transformation = np.matmul(subject_inv_transformation, p2_transformation)
            # get relative position of p2 to p3
            p2_rel_pos = np.matmul(p3_to_p2_transformation, zero_pos)[:3]
            # get relative rotation of p2 to p3
            p2_rel_rot = np.matmul(p3_to_p2_transformation, unit_vector)[:3]

            # calculate transformation from subject (p3) to p4
            p3_to_p4_transformation = np.matmul(subject_inv_transformation, p4_transformation)
            # get relative position of p4 to p3
            p4_rel_pos = np.matmul(p3_to_p4_transformation, zero_pos)[:3]
            # get relative rotation of p4 to p3
            p4_rel_rot = np.matmul(p3_to_p4_transformation, unit_vector)[:3]

            normalized_data.append([p1_rel_pos[0], p1_rel_pos[1], p1_rel_pos[2], p1_rel_rot[0], p1_rel_rot[1], p1_rel_rot[2],
                                    p2_rel_pos[0], p2_rel_pos[1], p2_rel_pos[2], p2_rel_rot[0], p2_rel_rot[1], p2_rel_rot[2],
                                    p4_rel_pos[0], p4_rel_pos[1], p4_rel_pos[2], p4_rel_rot[0], p4_rel_rot[1], p4_rel_rot[2] ])
        else:
            # calculate transformation from subject (p4) to p1
            p4_to_p1_transformation = np.matmul(subject_inv_transformation, p1_transformation)
            # get relative position of p1 to p4
            p1_rel_pos = np.matmul(p4_to_p1_transformation, zero_pos)[:3]
            # get relative rotation of p1 to p4
            p1_rel_rot = np.matmul(p4_to_p1_transformation, unit_vector)[:3]

            # calculate transformation from subject (p4) to p2
            p4_to_p2_transformation = np.matmul(subject_inv_transformation, p2_transformation)
            # get relative position of p2 to p4
            p2_rel_pos = np.matmul(p4_to_p2_transformation, zero_pos)[:3]
            # get relative rotation of p2 to p4
            p2_rel_rot = np.matmul(p4_to_p2_transformation, unit_vector)[:3]

            # calculate transformation from subject (p4) to p3
            p4_to_p3_transformation = np.matmul(subject_inv_transformation, p3_transformation)
            # get relative position of p3 to p4
            p3_rel_pos = np.matmul(p4_to_p3_transformation, zero_pos)[:3]
            # get relative rotation of p3 to p4
            p3_rel_rot = np.matmul(p4_to_p3_transformation, unit_vector)[:3]

            normalized_data.append([p1_rel_pos[0], p1_rel_pos[1], p1_rel_pos[2], p1_rel_rot[0], p1_rel_rot[1], p1_rel_rot[2],
                                    p2_rel_pos[0], p2_rel_pos[1], p2_rel_pos[2], p2_rel_rot[0], p2_rel_rot[1], p2_rel_rot[2],
                                    p3_rel_pos[0], p3_rel_pos[1], p3_rel_pos[2], p3_rel_rot[0], p3_rel_rot[1], p3_rel_rot[2] ])

    return normalized_data

def main():
    if len(sys.argv) < 4:
        print("Error: The input arguments are not valid. Arg1: path to the csv file, Arg2: output path, Arg3: filename.")
        return
    
    path_to_source = sys.argv[1]
    subject = sys.argv[2]
    out_path = sys.argv[3]
    filename = sys.argv[4]

    # subject exp: p1
    print("subject:", subject)

    rows = read_csv(path_to_source)
    if len(rows) == 0:
        return
    
    normalized_data = normalize_data(rows, subject)
    save_csv(out_path, filename, normalized_data)

if __name__ == '__main__':
    '''
        This script calculates the normalized relative positions of the people to the 
        subject and saves the data to a csv file.
        Normalized => Rotate the coordinates so that the chest rotation of the subject
        is on the positive z-axis.
    '''
    main()