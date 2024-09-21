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
parser.add_argument('--num', type=int, default=2)
args = parser.parse_args()
input_size = args.input_size
bitsize = args.bitsize
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
alpha = args.alpha
margin = args.margin
num = args.num
print('num: ', num)

data_path = f"./descriptors_v2/descriptor_set_{num}_train_valid_test.mat" # Path to the Train_valid_test.mat file
data_path_valid = f"/scratch/p307791/descriptors/descriptor_set_{num}_train_valid.mat" # Path to the Train_valid.mat file
data_path_test = f"/scratch/p307791/descriptors/descriptor_set_{num}_train_test.mat" # Path to the Train_test.mat file

output_dir = f'final_model_selection_train_valid_test_v_{num}' 

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

#input_size	   output_size	learning_rate	batch_size	alpha	  margin	l1_reg	       l2_reg
#372	       48	        0.01	        32	        0.00001	  48	    0.000000e+00	1.000000e-08	90.22
epoch_values = [2000]
# learning_rate_values = [0.01]
# alphas = [0.00001]
# margin_values = [48]
# batch_size_values = [32]
# bitsize_values = [72]
# l1_reg_values = [0]
# l2_reg_values = [1e-08]

# Learning rate {0.1, 0.01}
# Bit size {16, 24, 32, 40, 48, 56, 64, 72}
# Batch size {32, 48, 64}
# Margin {24, 36, 48}
# Alpha {1e-3, 1e-5}
# L1 regularization {0, 1e-8}
# L2 regularization {0, 1e-8}

#List of Grid search parameters to iterate over

learning_rate_values = [ 0.1, 0.01]
alphas = [1e-03,  1e-5]
margin_values = [24, 36, 48] 
batch_size_values = [32, 48, 64]
bitsize_values = [16, 24, 32, 40, 48, 56, 64, 72]
l2_reg_values = [0, 1e-08]
l1_reg_values = [0, 1e-08]


def run():
    for epochs in epoch_values:
        for learning_rate in learning_rate_values:
            for alpha in alphas:
                for margin in margin_values:
                    for batch_size in batch_size_values:
                        for bitsize in bitsize_values:
                            for l1_reg in l1_reg_values:
                                for l2_reg in l2_reg_values:  
                                                           
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
                                    ax.set_xlabel('Value')
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
                                    for _,threshold in enumerate(thresholds_abs_values):
                                        
                                        mAP_valid_thresh = mAP_at_k_equals_R(tr_df = train_df, tst_or_vl_df = valid_df,thresh = threshold)                                       
                                        mAP_results_valid.append(mAP_valid_thresh)
                                    
                                    
                                    # Plotting                                  

                                    data_abs_values = {'mAP_valid': mAP_results_valid,
                                            'threshold': thresholds_abs_values}

                                    df_thresh_abs_values = pd.DataFrame(data_abs_values)
                                    df_thresh_abs_values.to_csv(output_dir +'/results_data_abs_values.csv',index = False)

                                    # Find the index of the maximum mAP value
                                    max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()

                                    # Retrieve the threshold corresponding to the maximum mAP
                                    threshold_max_map_abs_values = df_thresh_abs_values.loc[max_map_index, 'threshold']
                                    mAP_valid_abs_values = mAP_at_k_equals_R(tr_df = train_df, tst_or_vl_df = valid_df,thresh = threshold_max_map_abs_values)
                                  
                                    mAP_test_abs_values = mAP_at_k_equals_R(tr_df = train_df, tst_or_vl_df = test_df,thresh = threshold_max_map_abs_values)

                                    fig, ax = plt.subplots(figsize=(10/3, 3))

                                    plt.plot(thresholds_abs_values, mAP_results_valid,color='#ff7f0eff')
                                    plt.scatter(threshold_max_map_abs_values, mAP_valid_abs_values, color='#d62728ff', marker='o', s=25)
                                    plt.xlabel('Threshold')
                                    plt.ylabel('mAP')

                                    plt.rc('font', family='Nimbus Roman')

                                    plt.savefig(output_dir + '/Maps_curves_abs_values.svg',format='svg', dpi=1200)
                                    plt.show()

                                    #####################################################################
                                    #####################################################################


                                    # Create a dictionary with the variable names and their values
                                    data = {'optimal threshold': [threshold_max_map_abs_values],
                                            'Best Validation mAP': [mAP_valid_abs_values],
                                            'Test mAP': [mAP_test_abs_values]
                                            }

                                    # Create the DataFrame
                                    df_results = pd.DataFrame(data)
                                    df_results.to_csv(output_dir +'/final_results.csv',index = False)
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
                                        
                                    }

                                    results_df = pd.DataFrame([results])
                                    if not os.path.isfile(output_dir + "/model_selection_train_valid_and_test_30082024_top100_v3_layers.csv"):
                                        df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg', 'epochs',  'threshold_max_map_abs_values',  'mAP_valid_abs_values', 'mAP_test_abs_values'])
                                        results_df = pd.concat([df, results_df])
                                        results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_30082024_top100_v3_layers.csv", index=False)
                                    else:
                                        df = pd.read_csv(output_dir + "/model_selection_train_valid_and_test_30082024_top100_v3_layers.csv")
                                        results_df = pd.concat([df, results_df], ignore_index=True)
                                        results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_30082024_top100_v3_layers.csv", index=False)




if __name__ == '__main__' :
    run()

