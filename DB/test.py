'''
import easyocr
reader = easyocr.Reader(['ch_sim','en'], gpu=False) # this needs to run only once to load the model into memory
result = reader.readtext('/home/yuningc/jersey-2023/train_light/images/19/19_130.jpg',min_size = 3, text_threshold = 0.01, allowlist="0123456789", mag_ratio = 2.5)
print(result)

import os

base_label_p = os.path.join("/home/yuningc/roboflow/train","labels")
labels = os.listdir(base_label_p)
base = 0
print(len(labels))
for index in range(len(labels)):
    #base = 0
    with open(os.path.join(base_label_p,labels[index]), "r") as f:
        c = f.readlines()
        if len(c)>2:
            base+=1
        
        for i in c:
            if int(i[0])>10:
                print(i)
                base = base*10+int(i[0])
        
print(base)
'''

import os

def print_directory_files_to_text(directory, output_file):
    try:
        with open(output_file, 'w') as f:
            files = os.listdir(directory)
            for file in files:
                f.write(file + '\n')
        print("Directory files printed to '{}' successfully.".format(output_file))
    except Exception as e:
        print("Error:", e)

# Example usage:
directory = '/home/yuningc/DB/datasets/jersey_hello/test_images'
output_file = '/home/yuningc/DB/datasets/jersey_hello/test_list.txt'

print_directory_files_to_text(directory, output_file)
