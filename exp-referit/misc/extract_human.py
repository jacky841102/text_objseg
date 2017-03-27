import pickle
import os

saiapr_path = "../saiaprtc12ok/benchmark/saiapr_tc-12/"
save_path = "./exp-referit/data/human_list"
human_label_id = set({66, 88, 120, 122, 126, 187, 51, 52, 53, 135, 160, 
                    273, 12, 77, 13})
human_region_set = set()

for folder in os.listdir(saiapr_path):
    label_file = saiapr_path + folder + '/labels.txt'
    print("processing " + folder)
    try:
        with open(label_file, 'r') as file:
            for line in file:
                try:
                    image_name, region, id = line.split()    
                except ValueError:
                    continue
                if int(id) in human_label_id:
                    human_region_set.add(image_name + '_' + region)
    except IOError:
        continue
with open(save_path, 'w') as f:
    pickle.dump(human_region_set, f)
print('done!')



