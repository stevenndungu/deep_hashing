
#%%
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

#https://ntpuccw.blog/%E5%B0%8D%E5%BD%B1%E5%83%8F%E7%9F%A9%E9%99%A3%E5%81%9A-convolution-%E7%9A%84%E5%9F%BA%E6%9C%AC%E8%AA%8D%E8%AD%98%E8%88%87%E7%A4%BA%E7%AF%84/


# Set the dimensions of the image
width, height = 150, 150

# Generate a random image with Gaussian distribution (mean 0, sigma 1)
random_image = np.random.normal(loc=0, scale=1, size=(height, width))


# Convert the random image to a PIL Image
random_image_pil = Image.fromarray((255 * (random_image - np.min(random_image)) / (np.max(random_image) - np.min(random_image))).astype(np.uint8))

#blur_by_gaussian = Image.fromarray(np.uint8(cm.gist_earth(random_image)*255))

filtered_image = random_image_pil.filter(ImageFilter.GaussianBlur(radius=25))


blur_by_gaussian = gaussian_filter(random_image_pil, sigma = 25)


################################################################
################################################################

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))

ax1.imshow(random_image, cmap='gray')
ax1.set_title('Random Gaussian Image (mean=0, sigma=1)')


ax3.imshow(filtered_image, cmap='gray')
ax3.set_title('Resultant Gaussian noise image - V2')

ax2.imshow(np.array(blur_by_gaussian), cmap = 'gray')
ax2.set_title('Resultant Gaussian noise image - V1')

plt.tight_layout()
plt.show()



################################################################
################################################################

image = Image.open(r"I:\My Drive\deep_learning\deep_hashing\data_complete\train\Bent\J004151.59+002836.2.jpg")

# Create a figure with 1 row and 3 columns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# First subplot
ax1.imshow(np.array(image)[:,:,1], cmap='gray')
ax1.set_title('Original Image')

# Second subplot
ax2.imshow(np.array(image)[:,:,1] + np.array(blur_by_gaussian), cmap='gray')
ax2.set_title('Image with Gaussian perturbation - V1')

# Third subplot
ax3.imshow(np.array(image)[:,:,1] + np.array(filtered_image), cmap='gray')
ax3.set_title('Image with Gaussian perturbation - V2')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# %%


