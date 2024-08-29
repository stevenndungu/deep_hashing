
#%%
import numpy as np
from PIL import Image
import os
import pylab as plt

seed = 42
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def save_array_as_jpeg(array, destination_path):
    """
    Save a numpy array as a JPEG image.
    
    Parameters:
    array (numpy.ndarray): The input array to be saved as an image.
    destination_path (str): The file path where the JPEG will be saved.
    
    Returns:
    None
    """
    # Normalize the array to [0, 1] range
    array_normalized = (array - np.min(array)) / (np.max(array) - np.min(array))
    
    # Convert to 8-bit unsigned integers
    array_8bit = (array_normalized * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(array_8bit)
    
    # If the image is not in RGB mode, convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save the image
    img.save(destination_path)

def shifted_gaussian(mean1=0,mean2=0,fwhm=1,size=150,image_resolution = 1.28):
    std = fwhm/(2*np.sqrt(2*np.log(2)))
    std_pixels = std/image_resolution
    #print(std_pixels)
    x = np.linspace(-size/2,size/2,size)
    y = np.linspace(-size/2,size/2,size)
    xx,yy = np.meshgrid(x,y)
    zz = np.exp(-((xx-mean1)**2+(yy-mean2)**2)/(2*std_pixels**2))
    return zz

#fwhm_psf = 5''
def correlated_noise(img_dimension, std_deviation_gaus_noise, fwhm_psf, image_resolution, path):
    img_gauss_noise = np.random.normal(0, std_deviation_gaus_noise, (img_dimension,img_dimension))
    kernal = shifted_gaussian(mean1=0,mean2=0,fwhm=fwhm_psf,size=img_dimension,image_resolution = image_resolution)

    img_gauss_blurred_noise = np.zeros(img_gauss_noise.shape,dtype=float)

    for k in range(img_gauss_noise.shape[0]):
        #print(k)
        for j in range(img_gauss_noise.shape[1]):
            kernal_shifted = shifted_gaussian(mean1=k-img_dimension/2,mean2=j-img_dimension/2,fwhm=fwhm_psf,size=img_dimension,image_resolution = image_resolution)
            img_gauss_blurred_noise += kernal_shifted*img_gauss_noise[k,j]
   

    
    diff = np.array(img_gauss_blurred_noise).max() - np.array(img_gauss_blurred_noise).min()
    # Convert the random image to a PIL Image
    random_noise_image = Image.fromarray((diff * (img_gauss_blurred_noise - np.min(img_gauss_blurred_noise)) / (np.max(img_gauss_blurred_noise) - np.min(img_gauss_blurred_noise))).astype(np.uint8))

    random_noise_image = np.array(random_noise_image)/255

    image = Image.open(path)
    image = np.array(image)/255

    factors = [3,4]
    result = np.array(random_noise_image) + np.array(image)[:,:,1]
    result1 = np.array(random_noise_image)*factors[0] + np.array(image)[:,:,1]
    result2 = np.array(random_noise_image)*factors[1] + np.array(image)[:,:,1]
    
    return image, random_noise_image, result, result1, result2

############################################
### Image | result || result1 |+ result2 ###
############################################

def generate_noise(seed, path):
    set_seed(seed)
    image, random_noise_image, result, result1, result2 = correlated_noise(img_dimension=150, std_deviation_gaus_noise = 1.5, fwhm_psf = 5, image_resolution = 1.28, path = path)
  
    return image, random_noise_image, result, result1, result2

#%%
# # Specify the source and destination folders
data_complete = "I:/My Drive/deep_learning/deep_hashing/data_complete"
# data_gnoise_f1 = './data_gnoise_f1'
# data_gnoise_f3 = './data_gnoise_f3'
# data_gnoise_f5 = './data_gnoise_f5'

class_img = ['Bent', 'Compact', 'FRI', 'FRII']
folders = ['train', 'test', 'valid']

# for i,folder in enumerate(folders):
   
#    for _,class_im in enumerate(class_img):
      
#       source_folder = f'{data_complete}/{folder}/{class_im}/'
#       destination_folder_f1 = f'{data_gnoise_f1}/{class_im}/{folder}/'
#       destination_folder_f3 = f'{data_gnoise_f3}/{class_im}/{folder}/'
#       destination_folder_f5 = f'{data_gnoise_f5}/{class_im}/{folder}/'
            
#       # Get the list of files in the source folder
#       file_list = os.listdir(source_folder)

#       # Iterate over the files and move them one by one
#       for seed,file_name in enumerate(file_list):
         
#          # Generate the full path of the source and destination files
#          source_path = os.path.join(source_folder, file_name)
#          destination_path_f1 = os.path.join(destination_folder_f1, file_name)
#          destination_path_f3 = os.path.join(destination_folder_f3, file_name)
#          destination_path_f5 = os.path.join(destination_folder_f5, file_name)
         
#          # Move the file
#          #shutil.copy(source_path, destination_path)
#          # Call the function with the desired seed and path
#          image, _, result_f1, result_f3, result_f5  = generate_noise(seed = seed, path = source_path)


#          save_array_as_jpeg(result_f1, destination_path_f1)
#          save_array_as_jpeg(result_f3, destination_path_f3)
#          save_array_as_jpeg(result_f5, destination_path_f5)
         
#    print(f'Folder {folder} is complete ....')

#%%
folder = folders[0]
class_im = class_img[2]
source_folder = f'{data_complete}/{folder}/{class_im}/'
file_list = os.listdir(source_folder)
file_name = file_list[8]
source_path = os.path.join(source_folder, file_name)
#source_path = r'C:\Users\P307791\Documents\deep_hashing_github\data\train\Bent\J010241.68+005027.7.jpg'
image, _, result_f1, result_f3, result_f5  = generate_noise(seed = seed, path = source_path)
factors = [3,4]


fig, (ax0,ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 5))

im0 = ax0.imshow(np.array(image)[:,:,1],cmap='viridis')
ax0.set_title('Original image ')
#fig.colorbar(im0, ax=ax0)
ax0.axis('off') 


im1 = ax1.imshow(result_f1,cmap='viridis')
ax1.set_title(f'resultant image: factor {1}')
#fig.colorbar(im1, ax=ax1)
ax1.axis('off')  


im2 = ax2.imshow(result_f3,cmap='viridis')
ax2.set_title(f'resultant image: factor {factors[0]}')
#fig.colorbar(im2, ax=ax2)
ax2.axis('off') 

im3 = ax3.imshow(result_f5,cmap='viridis')
ax3.set_title(f'resultant image: factor {factors[1]}')
#fig.colorbar(im3, ax=ax3)
ax3.axis('off')  

plt.tight_layout()
plt.savefig('overall_noise_results/FRII_image1.png',format='png')
plt.savefig('overall_noise_results/FRII_image1.svg',format='svg', dpi=1200)
plt.show()

   




# %%
