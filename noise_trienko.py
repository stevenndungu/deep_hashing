
#%%
import numpy as np
import pylab as plt
import numpy as np
from PIL import Image
import os
import pylab as plt
seed = 42
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def shifted_gaussian(mean1=0,mean2=0,fwhm=1,size=128,image_resolution = 1.28):
    std = fwhm/(2*np.sqrt(2*np.log(2)))
    std_pixels = std/image_resolution
    #print(std_pixels)
    x = np.linspace(-size/2,size/2,size)
    y = np.linspace(-size/2,size/2,size)
    xx,yy = np.meshgrid(x,y)
    zz = np.exp(-((xx-mean1)**2+(yy-mean2)**2)/(2*std_pixels**2))
    return zz

#fwhm_psf = 5''
def correlated_noise(img_dimension=128, std_deviation_gaus_noise = 1e-6, fwhm_psf = 5, image_resolution = 1.28):
    img_gauss_noise = np.random.normal(0, std_deviation_gaus_noise, (img_dimension,img_dimension))
    plt.imshow(img_gauss_noise)
    plt.show()
    kernal = shifted_gaussian(mean1=0,mean2=0,fwhm=fwhm_psf,size=img_dimension,image_resolution = image_resolution)
    plt.imshow(kernal)
    plt.show()

    img_gauss_blurred_noise = np.zeros(img_gauss_noise.shape,dtype=float)

    for k in range(img_gauss_noise.shape[0]):
        #print(k)
        for j in range(img_gauss_noise.shape[1]):
            kernal_shifted = shifted_gaussian(mean1=k-img_dimension/2,mean2=j-img_dimension/2,fwhm=fwhm_psf,size=img_dimension,image_resolution = image_resolution)
            img_gauss_blurred_noise += kernal_shifted*img_gauss_noise[k,j]

    plt.imshow(img_gauss_blurred_noise)
    plt.show()    

    rms = np.sqrt(np.mean(img_gauss_blurred_noise**2))
    print(str(rms)+ " Jy/beam")



def main():
    correlated_noise(img_dimension=128, std_deviation_gaus_noise = 0.75e-5, fwhm_psf = 5, image_resolution = 1.28)
    pass


#if __name__ == "__main__":
main()
# %%

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

img_gauss_noise = np.random.normal(0, 2, (150,150))
plt.imshow(img_gauss_noise)
plt.show()


# %%

destination_path = 'C:/Users/P307791/Documents/deep_hashing_github/radio_v3'
source_path = 'C:/Users/P307791/Documents/deep_hashing_github/radio'

folders = ['Bent', 'Compact', 'FRI', 'FRII']
class_img = ['train', 'test', 'valid']

# class_img = ['Bent', 'Compact', 'FRI', 'FRII']
# folders = ['train', 'test', 'valid']

for i,folder in enumerate(folders):
   
   for _,class_im in enumerate(class_img):
    
     
                
      # Get the list of files in the source folder
      file_list = os.listdir(f'{source_path}/{folder}/{class_im}/')

      # Iterate over the files and move them one by one
      for seed,file_name in enumerate(file_list):
         
         # Generate the full path of the source and destination files
        
       
         
         
         # Move the file
         #shutil.copy(source_path, destination_path)
         # Call the function with the desired seed and path
         image = Image.open(f'{source_path}/{folder}/{class_im}/{file_name}' )
         image = np.array(image)/255
         set_seed(seed)
         img_gauss_noise = np.random.normal(0, 0.1, (150,150))
         result1 = img_gauss_noise + np.array(image)[:,:,1]       
         save_array_as_jpeg(result1, f'{destination_path}/{folder}/{class_im}/{file_name}')
       
         
   print(f'Folder {folder} is complete ....')
# %%
