import os, json

dir = "data/dbnet-cropped/test_light/images"
dir_1 = "data/dbnet-cropped/test_light"

len_1 = []
len_0 = []

with open()


for folder_name in os.listdir(dir):
	tmp_path = os.listdir(os.path.join(dir, folder_name))
	if len(tmp_path) == 0:
		len_0.append(folder_name)
	elif len(tmp_path) == 1:
		len_1.append(folder_name)
	else:
		continue

print(dir)
print()
print("empty folder: \n", len_0)
print()
print("len 1 folder: \n", len_1)
	


