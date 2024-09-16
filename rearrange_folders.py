import os
import shutil

#%%

for fl in [5]:

   # Specify the source and destination folders
   source_root_folder = f'./data_gnoise_f{fl}'
   destination_root_folder = f'./data_gnoise_f{fl}_dnt'
 

   class_img = ['Bent', 'Compact', 'FRI', 'FRII']
   main_folders = ['train', 'test', 'valid']


   for _,class_im in enumerate(class_img):
      
      for i,folder in enumerate(main_folders):
         
         source_folder = f'{source_root_folder}/{class_im}/{folder}/'
         destination_folder = f'{destination_root_folder}/{folder}/{class_im}/'
               
         # Get the list of files in the source folder
         file_list = os.listdir(source_folder)

         # Iterate over the files and move them one by one
         for _,file_name in enumerate(file_list):
            
            # Generate the full path of the source and destination files
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            
            # Move the file
            shutil.copy(source_file, destination_file)
            
   print(f'data_gnoise_f{fl} is Complete ....')

def sanity_check():
   for fl in [1,3,5]:
      tt1 = len(os.listdir(f'data_gnoise_f{fl}_dnt/test/Bent')) == 103
      tt2 = len(os.listdir(f'data_gnoise_f{fl}_dnt/test/Compact')) == 100
      tt3 = len(os.listdir(f'data_gnoise_f{fl}_dnt/test/FRI')) == 100
      tt4 = len(os.listdir(f'data_gnoise_f{fl}_dnt/test/FRII')) == 101
      if tt1 and tt2 and tt3 and tt4:
         print(f'Test folder for data_gnoise_f{fl} is great')
      else:
         raise Exception(f'Test folder for data_gnoise_f{fl} is incomplete!!')

      
      tt1 = len(os.listdir(f'data_gnoise_f{fl}_dnt/train/Bent')) == 305
      tt2 = len(os.listdir(f'data_gnoise_f{fl}_dnt/train/Compact')) == 226
      tt3 = len(os.listdir(f'data_gnoise_f{fl}_dnt/train/FRI')) == 215
      tt4 = len(os.listdir(f'data_gnoise_f{fl}_dnt/train/FRII')) == 434
      if tt1 and tt2 and tt3 and tt4:
         print(f'Train folder for data_gnoise_f{fl} is great')
      else:
         raise Exception(f'Test folder for data_gnoise_f{fl} is incomplete!!')

      tt1 = len(os.listdir(f'data_gnoise_f{fl}_dnt/valid/Bent')) == 100
      tt2 = len(os.listdir(f'data_gnoise_f{fl}_dnt/valid/Compact')) == 80
      tt3 = len(os.listdir(f'data_gnoise_f{fl}_dnt/valid/FRI')) == 74
      tt4 = len(os.listdir(f'data_gnoise_f{fl}_dnt/valid/FRII')) == 144
      if tt1 and tt2 and tt3 and tt4:
         print(f'Valid folder for data_gnoise_f{fl} is great')
      else:
         raise Exception(f'Test folder for data_gnoise_f{fl} is incomplete!!')
      print('##################################################')
      print('\n')
      print('##################################################')

sanity_check()