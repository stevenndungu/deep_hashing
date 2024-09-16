#%%
import os

folder = ['Bent', 'Compact', 'FRI', 'FRII']
class_img = ['train', 'test', 'valid']

# class_img = ['Bent', 'Compact', 'FRI', 'FRII']
# folder = ['train', 'test', 'valid']

folder_name = 'data_complete_gnoise_f05'
path = '.'

folder_path = os.path.join(path, folder_name)

# Create the main folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for class_im in class_img:
    class_path = os.path.join(folder_path, class_im)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    for flder in folder:
        subfolder_path = os.path.join(class_path, flder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)


