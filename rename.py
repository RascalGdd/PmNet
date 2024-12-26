import os

path = r'/home/diandian/Diandian/DD/LungSeg/data/frames'

for folder in os.listdir(path):
    for filename in os.listdir(os.path.join(path, folder)):
        new_num = str(int(filename.split('.')[0].split('_')[1]) - 1)
        new_filename = ((6 - len(new_num)) * '0' + new_num) + '.jpg'
        # print(new_filename)
        os.rename(os.path.join(path, folder, filename), os.path.join(path, folder, new_filename))

        # os.rename(os.path.join(path, folder, filename), os.path.join(path, folder, filename.split('_')[1]))