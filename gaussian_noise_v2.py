

###############################################
################################################
#%%
import numpy as np
import os, random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
seed = 100
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

###############################################

path = fr"I:\My Drive\deep_learning\deep_hashing\data_complete\train\Bent\J004151.59+002836.2.jpg"

def apply_gaussian_noise(mean_val=0, sigma_val=1,blur_sigma=25, path=path):
   # Set the dimensions of the image
   width, height = 150, 150

   # Generate a random image with Gaussian distribution (mean 0, sigma 1)
   random_image = np.random.normal(loc=mean_val, scale=sigma_val, size=(height, width))

   # Convert the random image to a PIL Image
   random_image_pil = Image.fromarray((255 * (random_image - np.min(random_image)) / (np.max(random_image) - np.min(random_image))).astype(np.uint8))

   filtered_image = random_image_pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

   image = Image.open(path)
   resultant_image = np.array(image)[:,:,1] + np.array(filtered_image)

   im = Image.fromarray(resultant_image)
   im.save(path.split('\\')[-1])

   return random_image, filtered_image, image, resultant_image

random_image, filtered_image, image, resultant_image = apply_gaussian_noise(mean_val=0, sigma_val=1,blur_sigma=25, path=path)


###############################################

fig, (ax1,ax3) = plt.subplots(1,2,figsize=(10,5))

ax1.imshow(random_image, cmap='gray')
ax1.set_title('Random Gaussian Image (mean=0, sigma=1)')
ax1.axis('off')

ax3.imshow(filtered_image, cmap='gray')
ax3.set_title('Resultant Gaussian noise image - V2')
ax3.axis('off')

plt.tight_layout()
plt.show()

###############################################

# Create a figure with 1 row and 3 columns
fig, (ax1,  ax3) = plt.subplots(1, 2, figsize=(10, 5))

# First subplot
ax1.imshow(np.array(image)[:,:,1], cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off') 


# Third subplot
ax3.imshow(resultant_image, cmap='gray')
ax3.set_title('Image with Gaussian perturbation - V2')
ax3.axis('off')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

#%%
