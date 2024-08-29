
#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

sns.set_style("white")

df = pd.read_csv('overall_noise_results/results_data_abs_values.csv')
max_map_index = df['mAP_valid'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map_abs_values = df.loc[max_map_index, 'threshold']
max_map = df.loc[max_map_index, 'mAP_valid']

SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)

fig, ax = plt.subplots(figsize=(10/3, 3))

plt.plot(df.threshold, df.mAP_valid,color='#ff7f0eff')
plt.scatter(threshold_max_map_abs_values, max_map, color='#d62728ff', marker='o', s=25)
plt.xlabel('Threshold')
plt.ylabel('mAP')

plt.rc('font', family='Nimbus Roman')

plt.savefig('overall_noise_results/flops_plot.svg',format='svg', dpi=1200)
plt.show()



# %%
import numpy as np
import re

def extract_numbers(data_list):
    data_list = data_list.split(' ')
    numbers = []
    pattern = r'-?\d+\.\d+'  # Regular expression pattern to match floating-point numbers
    
    for item in data_list:
        if item:  # Check if the item is not an empty string
            match = re.match(pattern, item)
            if match:
                numbers.append(float(match.group()))
    
    return np.array(numbers)



df_testing = pd.read_csv("overall_noise_results/test_df.csv")
df_training = pd.read_csv("overall_noise_results/train_df.csv")
valid_df = pd.read_csv("overall_noise_results/valid_df.csv")

# Example usage
input_list = df_testing['predictions'][0]
output_list = extract_numbers(input_list)
#print(df_testing['predictions'][0],output_list)

df_testing['predictions'] = df_testing.predictions.map(extract_numbers)
df_training['predictions'] = df_training.predictions.map(extract_numbers)
valid_df['predictions'] = valid_df.predictions.map(extract_numbers)


dic_labels_rev = { 2:'Bent',
                3:'Compact',
                  0:'FRI',
                  1: 'FRII'
              }

# %%
df_plot = df_testing 
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10/3, 3))

# Iterate over label_code values
for label_code in range(4):
    dff = df_plot.query(f'label_code == {label_code}')
    out_array_train = []
    dd_train = np.array([out_array_train.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
    out_array_train = np.array(out_array_train)
    
    # Plot the KDE curve with a hue
    sns.kdeplot(out_array_train, label=f'{dic_labels_rev[label_code]}', ax=ax)

# Set the x-axis limits
ax.set_xlim(-1, 1)

# Customize the plot
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True)
ax.legend()
plt.savefig('paper images/Density_plot_train.png')
plt.savefig('paper images/Density_plot_train.svg',format='svg', dpi=1200)
plt.close()
# %%
