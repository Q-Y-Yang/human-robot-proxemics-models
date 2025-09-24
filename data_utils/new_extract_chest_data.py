import csv
import sys
import os

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

def process_data(rows):
    '''
        Extract the chest position (x, z) and rotation (x, z) of the chest of p1
        and the chest position (x, z) of p2, p3, p4 from the csv file
    '''
    # at index 3, the labels are stored
    # at index 5, the position / rotation label is stored
    # at index 6, the coordinate axis is stored (X, Y, Z)
    # at index 7, the data starts

    # find the label
    # we assume the data consists of 4 people. Each has a Chest position and rotation
    # we only consider the X and Z component of the position / rotation
    # we need the chest position and rotation of p1
    # we need the chest position of p2
    # we need the chest position of p3
    # we need the chest position of p4

    label_row = rows[3]
    p1_chest_index = label_row.index('p1_Chest')
    p2_chest_index = label_row.index('p2_Chest')
    p3_chest_index = label_row.index('p3_Chest')
    p4_chest_index = label_row.index('p4_Chest')

    # the rotation columns start at the label index (X,Y,Z,W) 
    # the position columns start at the label index + 4 (X,Y,Z)
    # check if we ar right
    rot_pos_row = rows[5]
    coordinate_row = rows[6]

    # ----------- check if the data is structured as expected ------------
    if (rot_pos_row[p1_chest_index] != 'Rotation' or 
        rot_pos_row[p1_chest_index + 4] != 'Position' or
        coordinate_row[p1_chest_index + 4] != 'X'):
        print("Error: The data is not structured as expected, p1.")
        print("rot_pos_row[p1_chest_index]:", rot_pos_row[p1_chest_index])
        print("rot_pos_row[p1_chest_index + 3]:", rot_pos_row[p1_chest_index + 3])
        print("coordinate_row[p1_chest_index + 4]:", coordinate_row[p1_chest_index + 4])
        return
    
    if (rot_pos_row[p2_chest_index] != 'Rotation' or 
        rot_pos_row[p2_chest_index + 4] != 'Position' or
        coordinate_row[p2_chest_index + 4] != 'X'):
        print("Error: The data is not structured as expected, p2.")
        return
    
    if (rot_pos_row[p3_chest_index] != 'Rotation' or
        rot_pos_row[p3_chest_index + 4] != 'Position' or
        coordinate_row[p3_chest_index + 4] != 'X'):
        print("Error: The data is not structured as expected, p3.")
        return
    
    if (rot_pos_row[p4_chest_index] != 'Rotation' or
        rot_pos_row[p4_chest_index + 4] != 'Position' or
        coordinate_row[p4_chest_index + 4] != 'X'):
        print("Error: The data is not structured as expected, p4.")
        return
    
    # ----------- extract the data ------------
    # these will be 2d arrays
    p1_chest_pos = []
    p1_chest_rot = []
    p2_chest_pos = []
    p3_chest_pos = []
    p4_chest_pos = []

    # Pos: x => +4, z => +6
    # Rot: x => +0, z => +2
    i = 7
    for row in rows[7:]:
        # Sometimes the data has missing values
        # => Use try except to catch invalid values
        try:
            # get the x and z component of the chest position
            temp_p1_chest_pos = [float(row[p1_chest_index + 4]), float(row[p1_chest_index + 6])]
        except ValueError:
            print("Error: Invalid value for p1_chest_pos at ", p1_chest_index + 4, p1_chest_index + 6, " in row ", i)
            # use the last valid value. If there is no valid value, use (0, 0)
            if len(p1_chest_pos) > 0:
                temp_p1_chest_pos = p1_chest_pos[-1]
            else:
                temp_p1_chest_pos = [0, 0]

        try:
            temp_p1_chest_rot = [float(row[p1_chest_index]), float(row[p1_chest_index + 2])]
        except ValueError:
            print("Error: Invalid value for p1_chest_rot at ", p1_chest_index, p1_chest_index + 2, " in row ", i)
        
            # use the last valid value. If there is no valid value, use (0, 0)
            if len(p1_chest_rot) > 0:
                temp_p1_chest_rot = p1_chest_rot[-1]
            else:
                temp_p1_chest_rot = [0, 0]

        try:
            temp_p2_chest_pos = [float(row[p2_chest_index + 4]), float(row[p2_chest_index + 6])]
        except ValueError:
            print("Error: Invalid value for p2_chest_pos at ", p2_chest_index + 4, p2_chest_index + 6, " in row ", i)
            # use the last valid value. If there is no valid value, use (0, 0)
            if len(p2_chest_pos) > 0:
                temp_p2_chest_pos = p2_chest_pos[-1]
            else:
                temp_p2_chest_pos = [0, 0]

        try:
            temp_p3_chest_pos = [float(row[p3_chest_index + 4]), float(row[p3_chest_index + 6])]
        except ValueError:
            print("Error: Invalid value for p3_chest_pos at ", p3_chest_index + 4, p3_chest_index + 6, " in row ", i)
            # use the last valid value. If there is no valid value, use (0, 0)
            if len(p3_chest_pos) > 0:
                temp_p3_chest_pos = p3_chest_pos[-1]
            else:
                temp_p3_chest_pos = [0, 0]

        try:
            temp_p4_chest_pos = [float(row[p4_chest_index + 4]), float(row[p4_chest_index + 6])]
        except ValueError:
            print("Error: Invalid value for p4_chest_pos at ", p4_chest_index + 4, p4_chest_index + 6, " in row ", i)
            # use the last valid value. If there is no valid value, use (0, 0)
            if len(p4_chest_pos) > 0:
                temp_p4_chest_pos = p4_chest_pos[-1]
            else:
                temp_p4_chest_pos = [0, 0]
                

        p1_chest_pos.append(temp_p1_chest_pos)
        p1_chest_rot.append(temp_p1_chest_rot)
        p2_chest_pos.append(temp_p2_chest_pos)
        p3_chest_pos.append(temp_p3_chest_pos)
        p4_chest_pos.append(temp_p4_chest_pos)
        i += 1

    return p1_chest_pos, p1_chest_rot, p2_chest_pos, p3_chest_pos, p4_chest_pos

def new_process_data(rows):
    '''
    this function extracts the chest position (x, z) and rotation (x, z) for p1 to p4
    '''
    

    label_row = rows[3]
    p1_chest_index = label_row.index('p1_Chest')
    p2_chest_index = label_row.index('p2_Chest')
    p3_chest_index = label_row.index('p3_Chest')
    p4_chest_index = label_row.index('p4_Chest')    


    # ------- extract chest pos data -------
    p1_chest_pos = []
    p1_chest_rot = []
    p2_chest_pos = []
    p2_chest_rot = []
    p3_chest_pos = []
    p3_chest_rot = []
    p4_chest_pos = []
    p4_chest_rot = []

    # Pos: x => +4, z => +6
    # Rot: x => +0, z => +2
    i = 7
    for row in rows[7:]:
        try:
            temp_p1_chest_pos = get_from_row(row, p1_chest_index, 'pos')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            print("ValueError: p1_chest_pos at ", i)
            if len(p1_chest_pos) > 0:
                temp_p1_chest_pos = p1_chest_pos[-1]
            else:
                temp_p1_chest_pos = [0, 0, 0]
        
        try:
            temp_p1_chest_rot = get_from_row(row, p1_chest_index, 'rot')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            print("ValueError: p1_chest_rot at ", i)
            if len(p1_chest_rot) > 0:
                temp_p1_chest_rot = p1_chest_rot[-1]
            else:
                temp_p1_chest_rot = [0, 0, 0, 0]

        try:
            temp_p2_chest_pos = get_from_row(row, p2_chest_index, 'pos')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p2_chest_pos) > 0:
                temp_p2_chest_pos = p2_chest_pos[-1]
            else:
                temp_p2_chest_pos = [0, 0, 0]

        try:
            temp_p2_chest_rot = get_from_row(row, p2_chest_index, 'rot')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p2_chest_rot) > 0:
                temp_p2_chest_rot = p2_chest_rot[-1]
            else:
                temp_p2_chest_rot = [0, 0, 0, 0]

        try:
            temp_p3_chest_pos = get_from_row(row, p3_chest_index, 'pos')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p3_chest_pos) > 0:
                temp_p3_chest_pos = p3_chest_pos[-1]
            else:
                temp_p3_chest_pos = [0, 0, 0]

        try:
            temp_p3_chest_rot = get_from_row(row, p3_chest_index, 'rot')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p3_chest_rot) > 0:
                temp_p3_chest_rot = p3_chest_rot[-1]
            else:
                temp_p3_chest_rot = [0, 0, 0, 0]

        try:
            temp_p4_chest_pos = get_from_row(row, p4_chest_index, 'pos')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p4_chest_pos) > 0:
                temp_p4_chest_pos = p4_chest_pos[-1]
            else:
                temp_p4_chest_pos = [0, 0, 0]

        try:
            temp_p4_chest_rot = get_from_row(row, p4_chest_index, 'rot')
        except ValueError:
            # return the last valid value. If there is no valid value, use (0, 0)
            if len(p4_chest_rot) > 0:
                temp_p4_chest_rot = p4_chest_rot[-1]
            else:
                temp_p4_chest_rot = [0, 0, 0, 0]

        p1_chest_pos.append(temp_p1_chest_pos)
        p1_chest_rot.append(temp_p1_chest_rot)
        p2_chest_pos.append(temp_p2_chest_pos)
        p2_chest_rot.append(temp_p2_chest_rot)
        p3_chest_pos.append(temp_p3_chest_pos)
        p3_chest_rot.append(temp_p3_chest_rot)
        p4_chest_pos.append(temp_p4_chest_pos)
        p4_chest_rot.append(temp_p4_chest_rot)

        i += 1

    return p1_chest_pos, p1_chest_rot, p2_chest_pos, p2_chest_rot, p3_chest_pos, p3_chest_rot, p4_chest_pos, p4_chest_rot


def get_from_row(row, index, data_type):
    '''
    Get the pos x, z or rot x, z from the row
    Return the values as a list of floats => [float, float]
    '''
    # Pos: x => +4, y => +5, z => +6
    # Rot: x => +0, y => +1 z => +2, w => +3
    try:
        if data_type == 'pos':
            try:
                result = [float(row[index + 4]), float(row[index + 5]), float(row[index + 6])]
                return result
            except ValueError:
                print("Error: Invalid value for ", data_type, " at ", index)
                raise ValueError
        elif data_type == 'rot':
            try:
                result = [float(row[index]), float(row[index + 1]), float(row[index + 2]), float(row[index + 3])]
                return result
            except ValueError:
                print("Error: Invalid value for ", data_type, " at ", index)
                raise ValueError
        else:
            print("Error: Invalid data type.")
            return [0, 0]
    except ValueError:
        print("Error: Invalid value for ", data_type, " at ", index)
        raise ValueError


    
if __name__ == '__main__':
    # this script extracts the chest position (x, y, z) and rotation (x, y, z, w) for each person
    # person = p1, p2, p3, p4
    # example: python extract_chest_pos.py ../get_from_here/ ../store_here/ 1.csv
    if len(sys.argv) != 4:
        print("Usage: python extract_chest_pos.py in_path out_path filename")
        sys.exit(0)

    path2csv = sys.argv[1]
    out_path = sys.argv[2]
    filename = sys.argv[3]

    rows = read_csv(path2csv)
    p1_chest_pos, p1_chest_rot, p2_chest_pos, p2_chest_rot, p3_chest_pos, p3_chest_rot, p4_chest_pos, p4_chest_rot  = new_process_data(rows)
    if p1_chest_pos is None:
        print("Error: Could not extract the data.")
        sys.exit(0)

    # save the data
    data = []
    for i in range(len(p1_chest_pos)):
        temp = [p1_chest_pos[i][0], p1_chest_pos[i][1], p1_chest_pos[i][2], p1_chest_rot[i][0], p1_chest_rot[i][1], p1_chest_rot[i][2], p1_chest_rot[i][3],
                p2_chest_pos[i][0], p2_chest_pos[i][1], p2_chest_pos[i][2], p2_chest_rot[i][0], p2_chest_rot[i][1], p2_chest_rot[i][2], p2_chest_rot[i][3],
                p3_chest_pos[i][0], p3_chest_pos[i][1], p3_chest_pos[i][2], p3_chest_rot[i][0], p3_chest_rot[i][1], p3_chest_rot[i][2], p3_chest_rot[i][3],
                p4_chest_pos[i][0], p4_chest_pos[i][1], p4_chest_pos[i][2], p4_chest_rot[i][0], p4_chest_rot[i][1], p4_chest_rot[i][2], p4_chest_rot[i][3]]
        data.append(temp)

    save_csv(out_path, filename, data)
    