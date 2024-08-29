#%%
import os
#import shutil
import numpy as np
import os, random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
seed = 100
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


def apply_gaussian_noise(mean_val, sigma_val,blur_sigma, source_path,destination_path):
   # Set the dimensions of the image
   width, height = 150, 150

   # Generate a random image with Gaussian distribution (mean 0, sigma 1)
   random_image = np.random.normal(loc=mean_val, scale=sigma_val, size=(height, width))

   # Convert the random image to a PIL Image
   random_image_pil = Image.fromarray((255 * (random_image - np.min(random_image)) / (np.max(random_image) - np.min(random_image))).astype(np.uint8))

   filtered_image = random_image_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

   image = Image.open(source_path)
   resultant_image = np.array(image)[:,:,1] + np.array(filtered_image)

   im = Image.fromarray(resultant_image)
   im.save(destination_path)

   return random_image, filtered_image, image, resultant_image


#%%
# Specify the source and destination folders
data_complete = "I:/My Drive/deep_learning/deep_hashing/data_complete"
data_g_noise = './data_g_noise'

class_img = ['Bent', 'Compact', 'FRI', 'FRII']
folder = ['train', 'test', 'valid']

for i,flder in enumerate(folder):
   
   for _,class_im in enumerate(class_img):
      
      source_folder = f'{data_complete}/{flder}/{class_im}/'
      destination_folder = f'{data_g_noise}/{class_im}/{flder}/'
            
      # Get the list of files in the source folder
      file_list = os.listdir(source_folder)

      # Iterate over the files and move them one by one
      for _,file_name in enumerate(file_list):
         
         # Generate the full path of the source and destination files
         source_path = os.path.join(source_folder, file_name)
         destination_path = os.path.join(destination_folder, file_name)
         
         # Move the file
         #shutil.copy(source_path, destination_path)
         apply_gaussian_noise(mean_val=0, sigma_val=1,blur_sigma=25, source_path=source_path,destination_path=destination_path)
         
   print(f'Folder {flder} is complete ....')

#%%



# %%
