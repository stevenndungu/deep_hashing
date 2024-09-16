#%%
import os
import shutil

#%%
# Specify the source and destination folders
augmented_data = "/scratch/p307791/data_gnoise_f05_new"
radio_augmented_data = 'data_complete_gnoise_f05'

# class_img = ['Bent', 'Compact', 'FRI', 'FRII']
# folder = ['train', 'test', 'valid']

folder = ['Bent', 'Compact', 'FRI', 'FRII']
class_img = ['train', 'test', 'valid']

for i,flder in enumerate(folder):
   
   for _,class_im in enumerate(class_img):
      
      source_folder = f'{augmented_data}/{flder}/{class_im}/'
      destination_folder = f'{radio_augmented_data}/{class_im}/{flder}/'
            
      # Get the list of files in the source folder
      file_list = os.listdir(source_folder)

      # Iterate over the files and move them one by one
      for _,file_name in enumerate(file_list):
         
         # Generate the full path of the source and destination files
         source_file = os.path.join(source_folder, file_name)
         destination_file = os.path.join(destination_folder, file_name)
         
         # Move the file
         shutil.copy(source_file, destination_file)
      print(f'{flder} : {class_im} done ...')
   print(f'folder {flder} done ...')