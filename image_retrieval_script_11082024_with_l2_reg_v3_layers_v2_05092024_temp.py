#%%
from cosfire_workflow_utils import *

# Hyperparameters
parser = argparse.ArgumentParser(description='COSFIRENet Training and Evaluation')
parser.add_argument('--input_size', type=int, default=372, help='Input size of the Descriptors')
parser.add_argument('--bitsize', type=int, default=36, help='Output size of the COSFIRENet')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--alpha', type=int, default= 0.001)
parser.add_argument('--margin', type=int, default=36)
parser.add_argument('--num', type=int, default=21)
args = parser.parse_args()
input_size = args.input_size
bitsize = args.bitsize
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
alpha = args.alpha
margin = args.margin
num = args.num


'''
From Image_retrieval_comparison updated mwu - 03092024_top100_map.html
The best bit size is 72 based on validation results. Then, based on the 26 data sets with 72 bit size, we perform the Mann-Whitney U test to identify the significant hyperparameters. The results show that 11th descriptor is best, below is the hyperparameter set:

'input_size', 'output_size', 'learning_rate', 'batch_size',   'alpha',   'margin', 'l1_reg', 'l2_reg',

    372            72	          0.01	          64	       0.00001	    24	       0	   1e-8	

for better illustration use...
372,72,0.01,64,0.001,36,0.0,0.0
'''
num = 11
epochs = 2000
bitsize = 72
learning_rate = 0.01
batch_size = 64
alpha = 0.00001
margin = 24
l1_reg = 0
l2_reg = 1e-8	

print('Descriptor number: ', num)
data_path = f"./descriptors_v2/descriptor_set_{num}_train_valid_test.mat" # Path to the Train_valid_test.mat file
data_path_valid = f"/scratch/p307791/descriptors/descriptor_set_{num}_train_valid.mat" # Path to the Train_valid.mat file
data_path_test = f"/scratch/p307791/descriptors/descriptor_set_{num}_train_test.mat" # Path to the Train_test.mat file

output_dir = f'FINAL_Results_05092024' 

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print(f"The directory {output_dir} already exists.")

print(output_dir)

#model 3 layers
class CosfireNet(nn.Module):
    def __init__(self, input_size, bitsize, l1_reg, l2_reg):
        super(CosfireNet, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hd = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.BatchNorm1d(300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.Tanh(),
            nn.Linear(200, bitsize),
            nn.Tanh()
        )


    def forward(self, x):        
        regularization_loss = 0.0
        for param in self.hd.parameters():
            regularization_loss += torch.sum(torch.abs(param)) * self.l1_reg  # L1 regularization
            regularization_loss += torch.sum(param ** 2) * self.l2_reg  # L2 regularization
        return self.hd(x), regularization_loss


# Data
class CosfireDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(preprocessing.normalize(dataframe.iloc[:, :-1].values), dtype=torch.float32)
        self.labels = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



# loss function
def DSHLoss(u, y, alpha, margin):
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    y = y.int()
    # Create a duplicate y_label
    y = y.unsqueeze(1).expand(len(y),len(y))
    y_label = torch.ones_like(torch.empty(len(y), len(y)))
    y_label[y == y.t()] = 0
    dist = torch.cdist(u, u, p=2).pow(2)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)
    B1 = torch.norm(torch.abs(u) - 1, p=1, dim=1)
    B2 = B1.unsqueeze(1).expand(len(y), len(y))
    loss3 = (B2 + B2.t()) * alpha
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)
    return minibatch_loss


dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}

dic_labels_rev = { 2:'Bent',
                3:'Compact',
                  0:'FRI',
                  1: 'FRII'
              }



def model_validate_and_predict():

    # Train Valid & Test data
    train_df,valid_df,test_df = get_and_check_data_prev(data_path,data_path_valid,data_path_test,dic_labels)                                

    # DataLoader for training set
    train_dataset = CosfireDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # DataLoader for validation set
    valid_dataset = CosfireDataset(valid_df)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
                    
    model = CosfireNet(input_size, bitsize, l1_reg, l2_reg)
    
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    # Lists to store training and validation losses
    train_losses = []
    train_losses_eval = []
    
    val_losses = []
    # Variables to keep track of the best model
    
    # Train the loop
    for _ in tqdm(range(epochs), desc='Training Progress', leave=True):
        model.train()
        total_train_loss = 0.0
        for _, (train_inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_outputs, reg_loss = model(train_inputs)
            loss = DSHLoss(u = train_outputs, y=labels, alpha = alpha, margin = margin) + reg_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * train_inputs.size(0)
        scheduler.step()

        # Calculate average training loss
        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valid_dataloader:
                val_outputs, reg_loss = model(val_inputs)
                val_loss = DSHLoss(u = val_outputs, y=val_labels, alpha = alpha, margin = margin) + reg_loss
                total_val_loss += val_loss.item() * val_inputs.size(0)

        # Calculate average validation loss
        average_val_loss = total_val_loss / len(valid_dataloader)
        val_losses.append(average_val_loss)


    

    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(output_dir + '/Train_valid_curves.png')
    plt.close()

    #########################################################################
    #################     Evaluate                                                  
    ##########################################################################

    #model.eval()
    valid_dataset_eval = CosfireDataset(valid_df)
    valid_dataloader_eval = DataLoader(valid_dataset_eval, batch_size=batch_size, shuffle=False)
    
    #valid_dataloader_eval = torch.utils.data.DataLoader(valid_dataset_eval,      sampler=ImbalancedDatasetSampler(valid_dataset_eval),batch_size=batch_size)

    train_dataset_eval = CosfireDataset(train_df)
    train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=batch_size, shuffle=False)
    #train_dataloader_eval = torch.utils.data.DataLoader(train_dataset_eval,      sampler=ImbalancedDatasetSampler(train_dataset_eval),batch_size=batch_size)

    test_dataset = CosfireDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Lists to store predictions
    predictions = []

    # Perform predictions on the train set
    with torch.no_grad():
        for train_inputs, _ in tqdm(train_dataloader_eval, desc='Predicting', leave=True):
            train_outputs,_ = model(train_inputs)
            predictions.append(train_outputs.numpy())

    # Flatten the predictions
    flat_predictions_train = [item for sublist in predictions for item in sublist]
     
    
    # Append predictions to the df_train DataFrame
    train_df['predictions'] = flat_predictions_train
    train_df['label_name'] = train_df['label_code'].map(dic_labels_rev)

    train_df.to_csv(output_dir +'/train_df.csv',index = False)

    flat_predictions_train = np.array(flat_predictions_train)
    np.savetxt(output_dir + '/predictions_train.out', flat_predictions_train, delimiter=',')
    
    #################################################################
    
    predictions = []
    # Perform predictions on the valid set
    with torch.no_grad():
        for valid_inputs, _ in tqdm(valid_dataloader_eval, desc='Predicting', leave=True):
            valid_outputs,_ = model(valid_inputs)
            predictions.append(valid_outputs.numpy())
    # Flatten the predictions
    flat_predictions_valid = [item for sublist in predictions for item in sublist]
    # Append predictions to the valid_df DataFrame
    valid_df['predictions'] = flat_predictions_valid
    valid_df['label_name'] = valid_df['label_code'].map(dic_labels_rev)
    valid_df.to_csv(output_dir +'/valid_df.csv',index = False)

    flat_predictions_valid = np.array(flat_predictions_valid)
    np.savetxt(output_dir + '/predictions_valid.out', flat_predictions_valid, delimiter=',')


    #########################################################################
    ##Testing
    #########################################################################
    #Perform predictions on the testing set

    predictions_test = []
    with torch.no_grad():
        for test_inputs, _ in tqdm(test_dataloader, desc='Predicting', leave=True):
            test_outputs,_ = model(test_inputs)
            predictions_test.append(test_outputs.numpy())

    # Flatten the predictions
    flat_predictions_test = [item for sublist in predictions_test for item in sublist]

    # Append predictions to the test_df DataFrame
    test_df['predictions'] = flat_predictions_test
    test_df['label_name'] = test_df['label_code'].map(dic_labels_rev)
    test_df.to_csv(output_dir +'/test_df.csv',index = False)

    flat_predictions_test = np.array(flat_predictions_test)
    np.savetxt(output_dir + '/predictions_test.out', flat_predictions_test, delimiter=',')

    SMALL_SIZE = 7
    MEDIUM_SIZE = 7
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('font', family='Nimbus Roman')
    
    df_plot = train_df 
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
    ax.grid(False)
    ax.legend()
    plt.savefig(output_dir +'/Density_plot_train.png')
    plt.savefig(output_dir +'/Density_plot_train.svg',format='svg', dpi=1200)
    plt.close()


    df_plot = test_df 
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
    ax.grid(False)
    ax.legend()
    plt.savefig(output_dir +'/Density_plot_test.png')
    plt.savefig(output_dir +'/Density_plot_test.svg',format='svg', dpi=1200)
    plt.close()


    df_plot = valid_df 
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
    ax.set_xlabel('Network outputs')
    ax.set_ylabel('Density')
    ax.grid(False)
    ax.legend()
    plt.savefig(output_dir +'/Density_plot_valid.png')
    plt.savefig(output_dir +'/Density_plot_valid.svg',format='svg', dpi=1200)
    plt.close()
    #####################################################################
    #####################################################################

    thresholds_abs_values = np.arange(-1, 1.2, 0.1)
    mAP_results_valid = []
    mAP_results_test = []
    
    for _,thresh in enumerate(thresholds_abs_values):
        
        mAP_valid_thresh,_, _, _, _,_,_ = mAP_values(train_df,valid_df,thresh = thresh, percentile = False)
        mAP_test_thresh,_, _, _, _,_,_ = mAP_values(train_df, test_df,thresh = thresh, percentile = False)

               

        mAP_results_valid.append(mAP_valid_thresh)
        mAP_results_test.append(mAP_test_thresh)
        

    # Plotting
    
    data_abs_values = {'mAP_valid': mAP_results_valid,
            'mAP_test': mAP_results_test,
            'threshold': thresholds_abs_values}

    df_thresh_abs_values = pd.DataFrame(data_abs_values)
    df_thresh_abs_values.to_csv(output_dir +'/results_data_abs_values.csv',index = False)

    # Find the index of the maximum mAP value
    max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()

    # Retrieve the threshold corresponding to the maximum mAP
    # threshold_max_map_abs_values = df_thresh_abs_values.loc[max_map_index, 'threshold']
    # mAP_valid_abs_values,train_binary, train_label, test_binary, valid_label = mAP_values(train_df,valid_df,thresh = threshold_max_map_abs_values, percentile = False)
    

    threshold_max_map_abs_values = df_thresh_abs_values.loc[max_map_index, 'threshold']
    
    mAP_valid_abs_values,_,_, _, _, _, _ = mAP_values(train_df, valid_df,thresh = threshold_max_map_abs_values, percentile = False)

    mAP_test_abs_values,mAP_std,mAP_values1, train_binary, train_label, test_binary, test_label = mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False)

    mAP_at_k_equals_R_res = mAP_at_k_equals_R(tr_df = train_df, tst_or_vl_df = test_df,thresh = threshold_max_map_abs_values)
    print('COSFIRE average mAP when R=K: ', mAP_at_k_equals_R_res)

        

    fig, ax = plt.subplots(figsize=(10/3, 3))

    plt.plot(thresholds_abs_values, mAP_results_valid,color='#ff7f0eff')
    plt.scatter(threshold_max_map_abs_values, mAP_valid_abs_values, color='#d62728ff', marker='o', s=25)
    plt.xlabel('Threshold')
    plt.ylabel('mAP')

    plt.rc('font', family='Nimbus Roman')

    plt.savefig(output_dir + '/Maps_curves_abs_values.svg',format='svg', dpi=1200)
    plt.savefig(output_dir + '/Maps_curves_abs_values.png')
    plt.close()

 
    #####################################################################
    #####################################################################
  

    
    # Create a dictionary with the variable names and their values
    data = {'optimal threshold': [threshold_max_map_abs_values],
            'Best Validation mAP': [mAP_valid_abs_values],
            'Test mAP': [mAP_test_abs_values],
            'Average mAP when R=K' : [mAP_at_k_equals_R_res]
            }

    # Create the DataFrame
    df_results = pd.DataFrame(data)
    df_results.to_csv(output_dir +'/final_results.csv',index = False)



    # def add_data_paths(train_paths = '/scratch/p307791/data_complete/train/*/*', dic_labels = dic_labels):
    #     df_labels_train_paths = pd.DataFrame()
    #     df_labels_train_paths['paths'] = glob.glob(train_paths)
    #     df_labels_train_paths['label'] = df_labels_train_paths['paths'].apply(lambda x: x.split(os.path.sep)[5] )
    #     df_labels_train_paths['label_code'] = df_labels_train_paths['label'].map(dic_labels)
    #     df_labels_train_paths = df_labels_train_paths.sort_values('label_code')
    #     df_labels_train_paths = df_labels_train_paths.reset_index()[['paths', 'label', 'label_code']]
    #     return df_labels_train_paths

    # df_labels_train_paths = add_data_paths(train_paths = '/scratch/p307791/data_complete/train/*/*', dic_labels = dic_labels)
    # df_labels_test_paths = add_data_paths(train_paths = '/scratch/p307791/data_complete/test/*/*', dic_labels = dic_labels)

    # def perf_percentages(input_data):
    #     unique, counts = np.unique(input_data, return_counts=True)
    #     df = pd.DataFrame()
    #     df['unique'] = unique
    #     df['counts'] = counts
    #     df['Percentage'] = np.round(counts / counts.sum() * 100)
    #     return df

            
    # def query_image(test_image_index = 190, 
    #             test_images_paths = df_labels_test_paths,
    #             train_images_db_paths = df_labels_train_paths,
    #             train_images_db = train_binary,
    #             test_binary = test_binary):

            
    #     print('Test Image is: ', test_images_paths.label[test_image_index])
    #     fig = plt.figure(figsize=(3, 3))
    #     image_test = Image.open(test_images_paths.paths[test_image_index])
    #     image_test = torch.from_numpy(np.array(image_test))
    #     plt.imshow(image_test[:, :, 1], cmap='viridis')
    #     plt.axis('off')
    #     plt.savefig(output_dir +f'/imr_image_query_{test_image_index}.png')
    #     plt.savefig(output_dir +f'/imr_image_query_{test_image_index}.svg',format='svg', dpi=1200)

    #     test_image = test_binary[test_image_index]  
    #     #np.count_nonzero(np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1,
    #     # 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])==np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    #     # 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]))
    #     similarity_distance = np.count_nonzero(test_image != train_images_db, axis=1)
    #     sort_indices = np.argsort(similarity_distance)
    #     top_indices = sort_indices[:100]
    #     #print(top_indices)
    #     paths_to_imgs = [train_images_db_paths.paths[index] for _,index in enumerate(top_indices)]
    #     df = perf_percentages([train_images_db_paths.label[index] for index in top_indices])
    #     print(df)
    #     cols = 7
    #     rows = 4

    #     fig = plt.figure(figsize=(2 * cols, 2 * rows))
    #     for col in range(cols):
    #         for i, img_path in enumerate(paths_to_imgs[:cols*rows]):
    #             ax = fig.add_subplot(rows, cols, i + 1)
    #             ax.grid(visible=False)
    #             ax.axis("off")
    #             image = Image.open(img_path)
    #             image = torch.from_numpy(np.array(image))
    #             ax.imshow(image[:, :, 1], cmap='viridis')
    #             ax.set_title(img_path.split(os.path.sep)[5])

    #     plt.savefig(output_dir +f'/imr_images_rb_{test_image_index}.png')
    #     plt.savefig(output_dir +f'/imr_images_rb_{test_image_index}.svg',format='svg', dpi=1200)
    #     #plt.close()
    
    # query_image(test_image_index = 15)
    # query_image(test_image_index = 150)
    # query_image(test_image_index = 350)
    # query_image(test_image_index = 270)
    # query_image(test_image_index = 251)
    # query_image(test_image_index = 271)
    # query_image(test_image_index = 200)

    results = {
        'input_size': input_size,
        'output_size': bitsize,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'alpha': alpha,                                        
        "margin": margin, 
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        'epochs': epochs,
        'threshold_max_map_abs_values': threshold_max_map_abs_values,
        'mAP_valid_abs_values': mAP_valid_abs_values,
        'mAP_test_abs_values': mAP_test_abs_values,
        'model_type': 'model_v3_layers',                              
        'data_path': data_path.split('/')[2]
    }

    results_df = pd.DataFrame([results])
    if not os.path.isfile(output_dir + "/model_selection_train_valid_and_test_05092024_v3_layers.csv"):
        df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg', 'epochs',  'threshold_max_map_abs_values',  'mAP_valid_abs_values', 'mAP_test_abs_values', 'model_type', 'data_path'])
        results_df = pd.concat([df, results_df])
        results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_05092024_v3_layers.csv", index=False)
    else:
        df = pd.read_csv(output_dir + "/model_selection_train_valid_and_test_05092024_v3_layers.csv")
        results_df = pd.concat([df, results_df], ignore_index=True)
        results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_05092024_v3_layers.csv", index=False)


    

    # pr_data = {
    #     "P": cum_recall.tolist(),
    #     "R": cum_prec.tolist()
    #     }

    # with open(output_dir + f'/COSFIRE_{num}_raw1.json', 'w') as f:
    #     f.write(json.dumps(pr_data))


    # num_dataset=train_df.shape[0]
    # index_range = num_dataset // 100
    # index = [i * 100 - 1 for i in range(1, index_range + 1)]
    # max_index = max(index)
    # overflow = num_dataset - index_range * 100
    # index = index + [max_index + i for i in range(1, overflow + 1)]
    # c_prec = cum_prec[index]
    # c_recall = cum_recall[index]

    # pr_data = {
    # "index": index,
    # "P": c_prec.tolist(),
    # "R": c_recall.tolist()
    # }

    # with open(output_dir + f'/COSFIRE_{num}.json', 'w') as f:
    #             f.write(json.dumps(pr_data))
    # pr_data = {
    #     "COSFIRE": output_dir + f'/COSFIRE_{num}.json',
    #     "DenseNet": '/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/DenseNet.json'
    # }
    # N = 200
    # # N = -1
    # for key in pr_data:
    #     path = pr_data[key]
    #     pr_data[key] = json.load(open(path))


    # # markers = "DdsPvo*xH1234h"
    # markers = ".........................."
    # method2marker = {}
    # i = 0
    # for method in pr_data:
    #     method2marker[method] = markers[i]
    #     i += 1

    # plt.figure(figsize=(15, 5))
    # plt.subplot(131)

    # for method in pr_data:
    #     P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    #     print(len(P))
    #     print(len(R))
    #     plt.plot(R, P, linestyle="-",  label=method)
    # plt.grid(False)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.legend()
    # plt.subplot(132)
    # for method in pr_data:
    #     P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    #     plt.plot(draw_range, R, linestyle="-",  label=method)
    # plt.xlim(0, max(draw_range))
    # plt.grid(False)
    # plt.xlabel('The number of retrieved samples')
    # plt.ylabel('recall')
    # plt.legend()

    # plt.subplot(133)
    # for method in pr_data:
    #     P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    #     plt.plot(draw_range, P, linestyle="-",  label=method)
    # plt.xlim(0, max(draw_range))
    # plt.grid(False)
    # plt.xlabel('The number of retrieved samples')
    # plt.ylabel('precision')
    # plt.legend()
    # plt.savefig(output_dir + f"/pr_{num}.png")
    # plt.savefig(output_dir + f"/pr_{num}.svg")



    # data_train = pd.read_csv("/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_training.csv", sep = ',')
    # data_test = pd.read_csv("/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_testing.csv", sep = ',' )

    # distances_densenet_dict = {}
    # distances_norm_densenet_dict = {}
    # distances_cosfire_dict = {}
    # distances_norm_cosfire_dict = {}

    # # Use tqdm to create a progress bar for the loop
    # for i in tqdm(range(test_df.shape[0]), desc="Processing Test Samples"):
    #     distances_densenet = []
    #     distances_norm_densenet = []
    #     distances_cosfire = []
    #     distances_norm_cosfire = []
    #     vector_a = data_test.predictions[i]
    #     vector_a_cosfire = test_df.predictions[i]
    #     for j in range(train_df.shape[0]):
    #         vector_b = data_train.predictions[j]
    #         distances_densenet.append(calculate_cosine_similarity(vector_a, vector_b))
    #         distances_norm_densenet.append(calculate_norm_distance(vector_a, vector_b))

    #         vector_b_cosfire = train_df.predictions[j]
    #         distances_cosfire.append(calculate_cosine_similarity_v2(vector_a_cosfire, vector_b_cosfire))
    #         distances_norm_cosfire.append(calculate_norm_distance_v2(vector_a_cosfire, vector_b_cosfire))
            

    #     # Convert lists to numpy arrays
    #     distances_densenet = np.array(distances_densenet)
    #     distances_norm_densenet = np.array(distances_norm_densenet)

    #     # Sort distances and store in dictionaries
    #     ind = np.argsort(distances_densenet)
    #     distances_densenet = distances_densenet[ind]
    #     distances_densenet_dict[i] = distances_densenet

    #     ind = np.argsort(distances_norm_densenet)
    #     distances_norm_densenet = distances_norm_densenet[ind]
    #     distances_norm_densenet_dict[i] = distances_norm_densenet

    #     # Convert lists to numpy arrays
    #     distances_cosfire = np.array(distances_cosfire)
    #     distances_norm_cosfire = np.array(distances_norm_cosfire)

    #     # Sort distances and store in dictionaries
    #     ind = np.argsort(distances_cosfire)
    #     distances_cosfire = distances_cosfire[ind]
    #     distances_cosfire_dict[i] = distances_cosfire

    #     ind = np.argsort(distances_norm_cosfire)
    #     distances_norm_cosfire = distances_norm_cosfire[ind]
    #     distances_norm_cosfire_dict[i] = distances_norm_cosfire


    # distances_norm_densenet_dict_list={}
    # distances_norm_cosfire_dict_list = {}
    # for i in range(len(distances_norm_densenet_dict.keys())):
    #     distances_norm_densenet_dict_list[i] = list(distances_norm_densenet_dict[i])  
    #     distances_norm_cosfire_dict_list[i] = list(distances_norm_cosfire_dict[i])


    # # Apply the conversion function to the dictionary
    # distances_norm_cosfire_dict_list = convert_to_serializable(distances_norm_cosfire_dict_list)
    # distances_norm_densenet_dict_list = convert_to_serializable(distances_norm_densenet_dict_list)
    # with open(output_dir + '/distances_norm_densenet_dict_list.json', 'w') as json_file:
    #     json.dump(distances_norm_densenet_dict_list, json_file)

    # with open(output_dir + '/distances_norm_cosfire_dict_list.json', 'w') as json_file:
    #     json.dump(distances_norm_cosfire_dict_list, json_file)

    # dat_norm_cosfire = distances_norm_cosfire_dict[0]
    # dat_norm_densenet = distances_norm_densenet_dict[0]
    # for i in range(1,len(distances_norm_cosfire_dict.keys())):
    #     dat_norm_cosfire =+ distances_norm_cosfire_dict[i]
    #     dat_norm_densenet =+ distances_norm_densenet_dict[i]
        
    # dat_cosfire_norm_average = dat_norm_cosfire/len(distances_norm_cosfire_dict.keys()) 
    # dat_densenet_norm_average = dat_norm_densenet/len(distances_norm_cosfire_dict.keys()) 


    
    # topk = 200
    # data1 = normalize_distance(dat_cosfire_norm_average[0:topk])

    # data2 = normalize_distance(dat_densenet_norm_average[0:topk])
    # # Create indices
    # indices = np.arange(len(data1))

    # # Create the plot
    # plt.figure(figsize=(12, 6))

    # # Plot both curves
    # plt.plot(indices, data1, label='Cosfire norm', marker='o')
    # plt.plot(indices, data2, label='DenseNet norm', marker='s')

    # # Customize the plot
    # plt.xlabel(f'Top {topk} images', fontsize=12)
    # plt.ylabel('Normalised distance', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.7)

    # # Use scientific notation on y-axis
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # # Show the plot
    # plt.tight_layout()
    # plt.savefig(output_dir + "/distance_measure_norm_comparison.png")


    # dat_cosfire = distances_cosfire_dict[0]
    # dat_densenet = distances_densenet_dict[0]
    # for i in range(1,len(distances_cosfire_dict.keys())):
    #     dat_cosfire =+ distances_cosfire_dict[i]
    #     dat_densenet =+ distances_densenet_dict[i]
        
    # dat_cosfire_average = dat_cosfire/len(distances_cosfire_dict.keys())
    # dat_densenet_average = dat_densenet/len(distances_densenet_dict.keys())

    # # Data vectors

    # data1 = normalize_distance(dat_cosfire_average[0:topk])
    # data2 = normalize_distance(dat_densenet_average[0:topk])

    # # Create indices
    # indices = np.arange(len(data1))

    # # Create the plot
    # plt.figure(figsize=(12, 6))

    # # Plot both curves
    # plt.plot(indices, data1, label='Cosfire cosine', marker='o')
    # plt.plot(indices, data2, label='DenseNet cosine', marker='s')

    # # Customize the plot
    # plt.xlabel(f'Top {topk} images', fontsize=12)
    # plt.ylabel('Normalised distance', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.7)

    # # Use scientific notation on y-axis
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # # Show the plot
    # plt.tight_layout()
    # plt.close()
    # plt.savefig(output_dir + "/distance_measure_cosine_comparison.png")




    # #distances_hamm_densenet_dict = {}
    # distances_hamm_cosfire_dict = {}

    # # Use tqdm to create a progress bar for the loop
    # for i in tqdm(range(test_binary.shape[0]), desc="Processing Test Samples"):
    #     distances_hamm_cosfire = []
    #     vector_a = test_binary[i]
    #     for j in range(train_binary.shape[0]):
    #         vector_b = train_binary[j]
    #         distances_hamm_cosfire.append(similarity_distance_count(vector_a, vector_b))

    #     distances_hamm_cosfire = np.array(distances_hamm_cosfire)
        
    #     ind = np.argsort(distances_hamm_cosfire)
    #     distances_hamm_cosfire = distances_hamm_cosfire[ind]
    #     distances_hamm_cosfire_dict[i] = distances_hamm_cosfire


    #     distances_hamm_cosfire_dict_list = {}
    #     for i in range(len(distances_hamm_cosfire_dict.keys())):

    #         distances_hamm_cosfire_dict_list[i] = list(distances_hamm_cosfire_dict[i])


        # Apply the conversion function to the entire dictionary
        #converted_dict = convert_to_serializable(distances_hamm_cosfire_dict_list)

        # Now save the converted dictionary as JSON
        # with open(output_dir + '/distances_hamm_cosfire_dict_list.json', 'w') as json_file:
        #     json.dump(distances_hamm_cosfire_dict_list, json_file)

        # with open(output_dir + '/distances_hamm_cosfire_dict_list.txt', 'w') as txt_file:
        #     txt_file.write(str(distances_hamm_cosfire_dict_list))



        # dat_hamm_cosfire = np.array(distances_hamm_cosfire_dict[0])

        # for i in range(1,len(distances_hamm_cosfire_dict.keys())):
        #     dat_hamm_cosfire = dat_hamm_cosfire + np.array(distances_hamm_cosfire_dict[i])
            
            
        # dat_cosfire_hamm_average = dat_hamm_cosfire/len(distances_hamm_cosfire_dict.keys()) 


        

        # topk = 100
        # data1 = dat_cosfire_hamm_average[0:topk]

        # #data2 = dat_densenet_hamm_average[0:topk]
        # # Create indices
        # indices = np.arange(len(data1))

        # # Create the plot
        # plt.figure(figsize=(12, 6))

        # # Plot both curves
        # plt.plot(indices, data1, label='Cosfire hamm', marker='o')
        # #plt.plot(indices, data2, label='DenseNet hamm', marker='s')

        # # Customize the plot
        # plt.xlabel(f'Top {topk} images', fontsize=12)
        # plt.ylabel('hammalised distance', fontsize=12)
        # plt.legend(fontsize=10)
        # plt.grid(True, linestyle='--', alpha=0.7)

        # # Use scientific notation on y-axis
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # # Show the plot
        # plt.tight_layout()
        # plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
        # plt.close()
   


        # dat_cosfire = distances_hamm_cosfire_dict[0]
        # #dat_densenet = distances_densenet_dict[0]
        # for i in range(1,len(distances_hamm_cosfire_dict.keys())):
        #     dat_cosfire =+ distances_hamm_cosfire_dict[i]
        #     #dat_densenet =+ distances_densenet_dict[i]
            
        # dat_cosfire_average = dat_cosfire/len(distances_hamm_cosfire_dict.keys())
        # #dat_densenet_average = dat_densenet/len(distances_densenet_dict.keys())

        # # Data vectors

        # data1 = dat_cosfire_average[0:topk]
        # #data2 = dat_densenet_average[0:topk]

        # # Create indices
        # indices = np.arange(len(data1))

        # # Create the plot
        # plt.figure(figsize=(12, 6))

        # # Plot both curves
        # plt.plot(indices, data1, label='Cosfire cosine', marker='o')
        # #plt.plot(indices, data2, label='DenseNet cosine', marker='s')

        # # Customize the plot
        # plt.xlabel(f'Top {topk} images', fontsize=12)
        # plt.ylabel('Hamming distance', fontsize=12)
        # plt.legend(fontsize=10)
        # plt.grid(True, linestyle='--', alpha=0.7)

        # # Use scientific notation on y-axis
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # # Show the plot
        # plt.tight_layout()       
        # plt.savefig(output_dir + "/distance_measure_hamm_comparison.png")
     
        # plt.close()

        
        # data_train = pd.read_csv("/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_training.csv", sep = ',')
        # data_test = pd.read_csv("/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_testing.csv", sep = ',' )

        # distances_densenet_dict = {}
        # distances_norm_densenet_dict = {}
        # distances_cosfire_dict = {}
        # distances_norm_cosfire_dict = {}

        # # Use tqdm to create a progress bar for the loop
        # for i in tqdm(range(test_df.shape[0]), desc="Processing Test Samples"):
        #     distances_densenet = []
        #     distances_norm_densenet = []
        #     distances_cosfire = []
        #     distances_norm_cosfire = []
        #     vector_a = data_test.predictions[i]
        #     vector_a_cosfire = test_df.predictions[i]
        #     for j in range(train_df.shape[0]):
        #         vector_b = data_train.predictions[j]
        #         distances_densenet.append(calculate_cosine_similarity(vector_a, vector_b))
        #         distances_norm_densenet.append(calculate_norm_distance(vector_a, vector_b))

        #         vector_b_cosfire = train_df.predictions[j]
        #         distances_cosfire.append(calculate_cosine_similarity_v2(vector_a_cosfire, vector_b_cosfire))
        #         distances_norm_cosfire.append(calculate_norm_distance_v2(vector_a_cosfire, vector_b_cosfire))
                

        #     # Convert lists to numpy arrays
        #     distances_densenet = np.array(distances_densenet)
        #     distances_norm_densenet = np.array(distances_norm_densenet)

        #     # Sort distances and store in dictionaries
        #     ind = np.argsort(distances_densenet)
        #     distances_densenet = distances_densenet[ind]
        #     distances_densenet_dict[i] = distances_densenet

        #     ind = np.argsort(distances_norm_densenet)
        #     distances_norm_densenet = distances_norm_densenet[ind]
        #     distances_norm_densenet_dict[i] = distances_norm_densenet

        #     # Convert lists to numpy arrays
        #     distances_cosfire = np.array(distances_cosfire)
        #     distances_norm_cosfire = np.array(distances_norm_cosfire)

        #     # Sort distances and store in dictionaries
        #     ind = np.argsort(distances_cosfire)
        #     distances_cosfire = distances_cosfire[ind]
        #     distances_cosfire_dict[i] = distances_cosfire

        #     ind = np.argsort(distances_norm_cosfire)
        #     distances_norm_cosfire = distances_norm_cosfire[ind]
        #     distances_norm_cosfire_dict[i] = distances_norm_cosfire


        # distances_norm_densenet_dict_list={}
        # distances_norm_cosfire_dict_list = {}
        # for i in range(len(distances_norm_densenet_dict.keys())):
        #     distances_norm_densenet_dict_list[i] = list(distances_norm_densenet_dict[i])  
        #     distances_norm_cosfire_dict_list[i] = list(distances_norm_cosfire_dict[i])


        # # Apply the conversion function to the dictionary
        # distances_norm_cosfire_dict_list = convert_to_serializable(distances_norm_cosfire_dict_list)
        # distances_norm_densenet_dict_list = convert_to_serializable(distances_norm_densenet_dict_list)
        # with open(output_dir + '/distances_norm_densenet_dict_list.json', 'w') as json_file:
        #     json.dump(distances_norm_densenet_dict_list, json_file)

        # with open(output_dir + '/distances_norm_cosfire_dict_list.json', 'w') as json_file:
        #     json.dump(distances_norm_cosfire_dict_list, json_file)


        # ########
        # distances_densenet_dict={}
        # distances_cosfire_dict_list = {}
        # for i in range(len(distances_norm_densenet_dict.keys())):
        #     distances_densenet_dict[i] = list(distances_norm_densenet_dict[i])  
        #     distances_cosfire_dict_list[i] = list(distances_cosfire_dict[i])


        # # Apply the conversion function to the dictionary
        # distances_cosfire_dict_list = convert_to_serializable(distances_cosfire_dict_list)
        # distances_densenet_dict = convert_to_serializable(distances_densenet_dict)
        # with open(output_dir + '/distances_cosine_densenet_dict.json', 'w') as json_file:
        #     json.dump(distances_densenet_dict, json_file)

        # with open(output_dir + '/distances_cosine_cosfire_dict_list.json', 'w') as json_file:
        #     json.dump(distances_cosfire_dict_list, json_file)

        


        
        label = 'mean_std'
        train_df_dn = pd.read_csv('/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_training.csv')
        test_df_dn = pd.read_csv('/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_testing.csv')
        valid_df_dn = pd.read_csv('/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_valid.csv')

        predictions_test = []
        predictions_train = []
        predictions_valid = []

        for i in range(train_df_dn.shape[0]):
            predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

        for i in range(test_df_dn.shape[0]):
            predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))
        
        for i in range(valid_df_dn.shape[0]):
            predictions_valid.append(np.array(ast.literal_eval(valid_df_dn.predictions[i])))

        train_df_dn['predictions'] = predictions_train
        test_df_dn['predictions'] = predictions_test
        valid_df_dn['predictions'] = predictions_valid

        thresholds_abs_values = np.arange(-1, 1.2, 0.1)
        mAP_results_valid = []
        mAP_results_test = []

        for _,thresh in enumerate(thresholds_abs_values):
            
            mAP_valid_thresh,_, _, _, _,_,_ = mAP_values(train_df_dn,valid_df_dn,thresh = thresh, percentile = False)  
            mAP_results_valid.append(mAP_valid_thresh)

        data_abs_values = {'mAP_valid': mAP_results_valid,        
                'threshold': thresholds_abs_values}

        df_thresh_abs_values = pd.DataFrame(data_abs_values)
        # Find the index of the maximum mAP value
        max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()
        threshold_max_map_abs_values_dn = df_thresh_abs_values.loc[max_map_index, 'threshold']


        mAP_at_k_equals_R_res_dn = mAP_at_k_equals_R(tr_df = train_df_dn, tst_or_vl_df = test_df_dn,thresh = threshold_max_map_abs_values_dn)
        print('DenseNet average mAP when R=K: ', mAP_at_k_equals_R_res_dn)

        topk=100
        mAP_dn,mAP_std_dn,mAP_values_dn, r_binary_dn, train_label_dn, q_binary_dn, valid_label_dn = mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
        print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

        topk_number_images_dn = list(range(50,225,25)) + [215]
        mAP_topk_dn = []
        mAP_topk_std_dn = []
        map_values_list_dn = []
        for _, topk in enumerate(topk_number_images_dn):
            maP_dn,mAP_std_dn,map_values_dn, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
            mAP_topk_dn.append(maP_dn)
            mAP_topk_std_dn.append(mAP_std_dn)
            map_values_list_dn.append(map_values_dn)
        

        data_densenet = {'topk_number_images': topk_number_images_dn,
            'mAP': mAP_topk_dn,
            'mAP_std': mAP_topk_std_dn}
        df_densenet = pd.DataFrame(data_densenet)
        df_densenet.to_csv(output_dir + f'/mAP_vs_{topk}_images_72bit_densenet_{label}.csv', index = False)


        topk_number_images_cosf = list(range(50,225,25)) + [215]
        mAP_topk_cosf = []
        mAP_topk_std_cosf = []
        map_values_list_cosf = []
        for _, topk in enumerate(topk_number_images_cosf):
            maP_cosf,mAP_std_cosf, map_values_cosf, _, _, _, _= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
            mAP_topk_cosf.append(maP_cosf)
            mAP_topk_std_cosf.append(mAP_std_cosf)
            map_values_list_cosf.append(map_values_cosf)

        data_cosf = {'topk_number_images': topk_number_images_cosf,
            'mAP': mAP_topk_cosf,
            'mAP_std': mAP_topk_std_cosf}
        df_cosfire = pd.DataFrame(data_cosf)
        df_cosfire.to_csv(output_dir + f'/mAP_vs_topk_images_72bit_cosfire_{label}.csv', index = False)


        fig, ax = plt.subplots(figsize=(10/3, 3))
        sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='COSFIRE',  marker='o')
        # plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
        sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  marker='D')
        # plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

        plt.xlabel('The number of retrieved samples')
        plt.ylabel('mAP')
        plt.ylim(80, 94)
        plt.savefig(output_dir + f'/map_vs_topk_number_images_densenet_{label}_cbd.png')
        plt.savefig(output_dir + f'/map_vs_topk_number_images_densenet_{label}_cbd.svg',format='svg', dpi=1200)
        plt.close()

        
        
         
        # distances_cosfire_dict_list = {}
        # for i in range(len(distances_norm_densenet_dict.keys())):             
        #     distances_cosfire_dict_list[i] = list(distances_cosfire_dict[i])


        # # Apply the conversion function to the dictionary
        # distances_cosfire_dict_list = convert_to_serializable(distances_cosfire_dict_list)
        # distances_densenet_dict = convert_to_serializable(distances_densenet_dict)
        # with open(output_dir + '/distances_cosine_densenet_dict.json', 'w') as json_file:
        #     json.dump(distances_densenet_dict, json_file)

        # with open(output_dir + '/distances_cosine_cosfire_dict_list.json', 'w') as json_file:
        #     json.dump(distances_cosfire_dict_list, json_file)

        distances_cosine_densenet_dict_list = json.load(open(output_dir + '/distances_cosine_densenet_dict.json'))
        distances_cosine_cosfire_dict_list = json.load(open(output_dir + '/distances_cosine_cosfire_dict_list.json'))


       
        #################################################
        #################################################
        #######         Bent class              #########
        #################################################
        #################################################
        # print('Class: ',test_df.lable_name[0])
        # print('Class: ',test_df.lable_name[102])
        topk=100
        start = 0
        end = 102
        query_number = 0

        dat_cos_densenet = np.array(distances_cosine_densenet_dict_list[str(start)])
        dat_cos_cosfire = np.array(distances_cosine_cosfire_dict_list[str(start)])
        for i in range(start,end):#len(distances_cosine_densenet_dict_list.keys())):
            dat_cos_densenet = dat_cos_densenet + np.array(distances_cosine_densenet_dict_list[str(i)])
            dat_cos_cosfire = dat_cos_cosfire + np.array(distances_cosine_cosfire_dict_list[str(i)])
            

        dat_cos_densenet_average = dat_cos_densenet/(end-start)#len(distances_cosine_cosfire_dict_list.keys())
        dat_cos_cosfire_average = dat_cos_cosfire/(end-start)#len(distances_cosine_densenet_dict_list.keys())



        mAP_dn,mAP_std,mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values_dn, percentile = False,topk=topk)
        print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

        mix_values_dn = np.array(range(1,101))
        only_correct_dn = pr_denom_dn[query_number]
        comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])

        irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if  value == 0]
        #print(irrelevant_images_dn)


        mAP_cosf,mAP_std,mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
        print('mAP top 100  for DenseNet is: ',np.round(mAP_cosf,2))


        mix_values_cosf = np.array(range(1,101))
        only_correct_cosf = pr_denom_cosf[query_number]
        comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])

        irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if  value == 0]
        #print(irrelevant_images_cosf)



        data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
        data2 = normalize_distance(dat_cos_densenet_average[0:topk])
        data3 = normalize_distance(distances_cosine_densenet_dict_list[str(query_number)][0:topk])
        data4 = normalize_distance(distances_cosine_cosfire_dict_list[str(query_number)][0:topk])

        # Create indices
        indices = np.arange(len(data1))

        # Create the plot
        plt.figure(figsize=(10/3, 3))

        # Plot both curves
        plt.plot(indices, data1, label='COSFIRE*')#, marker='o', markersize=3)
        plt.plot(indices, data2, label='DenseNet*')#, marker='s', markersize=3)
        plt.plot(indices, data3, label='DenseNet')#), marker='s', markersize=3)
        plt.plot(indices, data4, label='COSFIRE')#, marker='s', markersize=3)
        for xx in range(len(irrelevant_images_dn)):
            index = irrelevant_images_dn[xx][0]
            plt.scatter(index, data3[index], color='red', marker='o', s=30)

        for yy in range(len(irrelevant_images_cosf)):
            index = irrelevant_images_cosf[yy][0]
            plt.scatter(index, data4[index], color='black', marker='o', s=30)
        # Customize the plot
        plt.xlabel(f'Top {topk} images')
        plt.ylabel('Normalised distance')
        plt.legend()
        plt.grid(False)

        # Use scientific notation on y-axis
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Show the plot
        plt.tight_layout()
        plt.savefig(output_dir + '/distance_Bent.svg',format='svg', dpi=1200)
        plt.savefig(output_dir + '/distance_Bent.png')
        
        plt.close()


        def plot_distances(topk, start, end, label):

            # Initialize arrays for DenseNet and COSFIRE
            dat_cos_densenet = np.array(distances_cosine_densenet_dict_list[str(start)])
            dat_cos_cosfire = np.array(distances_cosine_cosfire_dict_list[str(start)])

            # Sum up the distances for the range from start to end
            for i in range(start, end):
                dat_cos_densenet += np.array(distances_cosine_densenet_dict_list[str(i)])
                dat_cos_cosfire += np.array(distances_cosine_cosfire_dict_list[str(i)])

            # Compute average distances
            dat_cos_densenet_average = dat_cos_densenet / (end - start)
            dat_cos_cosfire_average = dat_cos_cosfire / (end - start)

            # DenseNet mAP and performance
            mAP_dn, mAP_std, mAP_values1, pr_denom_dn, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df_dn, test_df_dn, thresh=threshold_max_map_abs_values_dn, percentile=False, topk=topk)
            print(f'mAP top {topk} for DenseNet is: {np.round(mAP_dn, 2)}')

            performance_dn = [len(pr_denom_dn[index][0]) for index in pr_denom_dn.keys()]
            performance_dn = np.array(performance_dn)
            print('DenseNet: ', (performance_dn.min(), performance_dn.max()))

            # Extract values greater than 90 and their respective indices for DenseNet
            # select_ploting_dn = [(i, value) for i, value in enumerate(performance_dn) if value >= 89]
            # df_plot_dn = pd.DataFrame(select_ploting_dn, columns=['query_number', 'value'])
            # df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
            # df_plot_dn = df_plot_dn.sort_values(['value'])
            # #query_number_dn = df_plot_dn.loc[df_plot_dn['value'].idxmin(), 'query_number']
            # # Find the first position where the value is less than 99

            for qn in range(20):

                df_plot_dn = pd.DataFrame({'query_number': list(pr_denom_dn.keys()),
                            'performance': performance_dn})
                df_plot_dn = df_plot_dn.query('performance <= 95')
                df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
                df_plot_dn = df_plot_dn.sort_values(['performance'],ascending=False)

                threshold_dn = 95
                # Use a while loop to adjust the threshold until df_plot_dn is not empty
                while df_plot_dn.empty:
                    # Update the select_ploting_dn list based on the current threshold
                    df_plot_dn = pd.DataFrame({'query_number': list(pr_denom_dn.keys()),
                            'performance': performance_dn})
                    df_plot_dn = df_plot_dn.query('performance <= @threshold_dn')
                    df_plot_dn = df_plot_dn.query('query_number >= @start and query_number <= @end')
                    df_plot_dn = df_plot_dn.sort_values(['performance'],ascending=False)
                    
                    # Reduce the threshold by 1
                    threshold_dn += 1

                # data_k = np.array((df_plot_dn.sort_values(['value']).value))
                # query_number_dn = np.where(data_k <= 100)[0][0]
                #print(df_plot_dn)
                query_number_dn = np.array(df_plot_dn.query_number)[qn]


                # COSFIRE mAP and performance
                mAP_cosf, mAP_std, mAP_values_cosf, pr_denom_cosf, r_binary, train_label, q_binary, valid_label = mAP_values_v2(train_df, test_df, thresh=threshold_max_map_abs_values, percentile=False, topk=topk)
                print(f'mAP top {topk} for COSFIRE is: {np.round(mAP_cosf, 2)}')

                performance_cosf = [len(pr_denom_cosf[index][0]) for index in pr_denom_cosf.keys()]
                performance_cosf = np.array(performance_cosf)
                print('COSFIRE: ', (performance_cosf.min(), performance_cosf.max()))

                # Extract values greater than 90 and their respective indices for COSFIRE
                #select_ploting_cosf = [(i, value) for i, value in enumerate(performance_cosf) if value >= 0]
                #df_plot_cosf = pd.DataFrame(select_ploting_cosf, columns=['query_number', 'value'])

                df_plot_cosf = pd.DataFrame({'query_number': list(pr_denom_cosf.keys()),
                            'performance': performance_cosf})
                df_plot_cosf = df_plot_cosf.query('performance <= 95')
                df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
                df_plot_cosf = df_plot_cosf.sort_values(['performance'],ascending=False)

                threshold_cosf = 95
                # Use a while loop to adjust the threshold until df_plot_cosf is not empty
                while df_plot_cosf.empty:
                    # Update the select_ploting_dn list based on the current threshold
                    df_plot_cosf = pd.DataFrame({'query_number': list(pr_denom_cosf.keys()),
                            'performance': performance_cosf})
                    df_plot_cosf = df_plot_cosf.query('performance <= @threshold_cosf')
                    df_plot_cosf = df_plot_cosf.query('query_number >= @start and query_number <= @end')
                    df_plot_cosf = df_plot_cosf.sort_values(['performance'],ascending=False)
                                
                    # Reduce the threshold by 1
                    threshold_cosf += 1


                #data_v = np.array((df_plot_cosf.sort_values(['value']).value))
                #query_number_cosf = np.where(data_v <= 100)[0][0]
                #print(df_plot_cosf)
                query_number_cosf = np.array(df_plot_cosf.query_number)[qn]


                # Normalize distances for plotting
                data1 = normalize_distance(dat_cos_cosfire_average[0:topk])
                data2 = normalize_distance(dat_cos_densenet_average[0:topk])

                distances_cosine_cosfire_dict_list_subset = {}
                distances_cosine_densenet_dict_list_subset = {}
                for i in range(start, end):
                    distances_cosine_cosfire_dict_list_subset[i] = np.array(distances_cosine_cosfire_dict_list[str(i)])
                    distances_cosine_densenet_dict_list_subset[i] = np.array(distances_cosine_densenet_dict_list[str(i)])

                data3 = normalize_distance(distances_cosine_densenet_dict_list_subset[query_number_dn][0:topk])
                data4 = normalize_distance(distances_cosine_cosfire_dict_list_subset[query_number_cosf][0:topk])

                # Create indices for plotting
                indices = np.arange(len(data1))

                # Create the plot
                plt.figure(figsize=(10/3, 3))

                # Plot both curves
                plt.plot(indices, data1, label='COSFIRE*')
                plt.plot(indices, data2, label='DenseNet*')
                plt.plot(indices, data3, label='DenseNet')
                plt.plot(indices, data4, label='COSFIRE')

                mix_values_cosf = np.array(range(1, topk + 1))
                only_correct_cosf = pr_denom_cosf[query_number_cosf]
                comb_predictions_cosf = np.array([1 if value in only_correct_cosf else 0 for value in mix_values_cosf])

                mix_values_dn = np.array(range(1, topk + 1))
                only_correct_dn = pr_denom_dn[query_number_dn]
                comb_predictions_dn = np.array([1 if value in only_correct_dn else 0 for value in mix_values_dn])

                # Mark irrelevant images on the plot
                irrelevant_images_dn = [(i, value) for i, value in enumerate(comb_predictions_dn) if value == 0]
                irrelevant_images_cosf = [(i, value) for i, value in enumerate(comb_predictions_cosf) if value == 0]
                print(f'{label}: ', irrelevant_images_cosf)
                for xx in range(len(irrelevant_images_dn)):
                    index = irrelevant_images_dn[xx][0]
                    plt.scatter(index, data3[index], color='red', marker='o', s=30)
                for yy in range(len(irrelevant_images_cosf)):
                    index = irrelevant_images_cosf[yy][0]
                    plt.scatter(index, data4[index], color='black', marker='o', s=30)

                # Customize the plot
                plt.xlabel(f'Top {topk} images')
                plt.ylabel('Normalized distance')
                plt.legend()
                plt.grid(False)

                # Use scientific notation on y-axis
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                # Show the plot
                plt.tight_layout()
                plt.savefig(output_dir + f'/distance_{label}_{qn}.svg', format='svg', dpi=1200)
                plt.savefig(output_dir + f'/distance_{label}_{qn}.png')
                plt.close()

                #stoping criteria
                print('df_plot_dn shape: ',df_plot_dn.shape[0])
                print('df_plot_cosf shape: ',df_plot_cosf.shape[0])
                if qn == np.array((df_plot_dn.shape[0],df_plot_cosf.shape[0])).min()-1:
                    break
                else:
                    
                    print('===================================')
                    print(f'============= {qn} ==========+====')
                    print('===================================')


        plot_distances(topk=100, start=0, end=102, label='Bent')
        plot_distances(topk=100, start=103, end=201, label='Compact')
        plot_distances(topk=100, start=203, end=301, label='FRI')
        plot_distances(topk=100, start=303, end=403, label='FRII')


if __name__ == '__main__' :
    model_validate_and_predict()

