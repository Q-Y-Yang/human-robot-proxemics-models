# this script uses other scripts 
import sys
import os

def new_extract():
    # extracts the chest position amd rotation of the chest for every person
    # p1_pos_x, p1_pos_y, p1_pos_z, p1_rot_x, p1_rot_y, p1_rot_z, p1_rot_w, ...
    in_path = "human_group_csv/"
    out_path = "all_chest_data/human_group/"
    # loop over all groups and all csv files
    for i in range(1, 11):
        for j in range(1, 43):
            file_path = in_path + "Group" + str(i) + "/" + str(j) + ".csv"
            if not os.path.exists(file_path):
                print("file not exists: " + file_path)
                continue
            out_path_temp = out_path + "Group" + str(i) + "/"
            if not os.path.exists(out_path_temp):
                os.makedirs(out_path_temp)
            out_file = str(j) + ".csv"
            os.system("python new_extract_chest_data.py " + file_path + " " + out_path_temp + " " + out_file)   

def new_calc_normalized():
    # calculates the normalized relative positions of the people
    in_path = "all_chest_data/human_group/"
    out_path = "all_rel_chest_data/human_group/"
    # loop over all groups and all csv files and all subjects
    for i in range(1, 11): 
        for j in range(1, 43):
            for subject in range(1, 5):
                file_path = in_path + "Group" + str(i) + "/" + str(j) + ".csv"
                if not os.path.exists(file_path):
                    print("file not exists: " + file_path)
                    continue
                out_path_temp = out_path + "Group" + str(i) + "/"
                if not os.path.exists(out_path_temp):
                    os.makedirs(out_path_temp)
                # changes the output file name depending on the subject
                out_file = str(j) + "_" + str(subject) + ".csv"
                os.system("python automate_normalized_chest_data.py " + file_path + " p" + str(subject) + " " +  out_path_temp + " " + out_file)
                

if __name__ == '__main__':
    # Beware: This script will run for a couple minutes and create about 62MB of data.
    # It is advised to run on comment out only one of the functions below at a time.
    new_extract()
    new_calc_normalized()
