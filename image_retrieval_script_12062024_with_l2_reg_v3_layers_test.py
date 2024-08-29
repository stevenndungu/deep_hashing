#%%
from cosfire_workflow_utils import *

num = 10
print('num: ', num)


data_path = f"descriptors_tain_valid_test/descriptor_set_{num}_train_valid_test.mat"
data_path_test = f"descriptors_classification/descriptor_set_{num}_train_test.mat"
data_path_valid = f"descriptors_classification/descriptor_set_{num}_train_valid.mat"

output_dir = f'final_model_selection_train_valid_test_v_{num}' 

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print(f"The directory {output_dir} already exists.")

print(output_dir)

#%%

# #model 1 layers
# class CosfireNet(nn.Module):
#     def __init__(self, input_size, bitsize, l1_reg, l2_reg):
#         super(CosfireNet, self).__init__()
#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg
#         self.hd = nn.Sequential(
#             nn.Linear(input_size, bitsize),
#             nn.Tanh()
#         )

# #model 2 layers
# class CosfireNet(nn.Module):
#     def __init__(self, input_size, bitsize, l1_reg, l2_reg):
#         super(CosfireNet, self).__init__()
#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg
#         self.hd = nn.Sequential(
#             nn.Linear(input_size, 300),
#             nn.BatchNorm1d(300),
#             nn.Tanh(),
#             nn.Linear(300, bitsize),
#             nn.Tanh()
#         )

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

# #model 4
# class CosfireNet(nn.Module):
#     def __init__(self, input_size, bitsize, l1_reg, l2_reg):
#         super(CosfireNet, self).__init__()
#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg
#         self.hd = nn.Sequential(
#             nn.Linear(input_size, 300),
#             nn.BatchNorm1d(300),
#             nn.Tanh(),
#             nn.Linear(300, 200),
#             nn.BatchNorm1d(200),
#             nn.Tanh(),
#             nn.Linear(200, 100),
#             nn.BatchNorm1d(100),
#             nn.Tanh(),
#             nn.Linear(100, bitsize),
#             nn.Tanh()
#         )
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

input_size = 372
epoch_values = [100]
learning_rate_values = [0.1]
alphas = [0.001]
margin_values = [48]
batch_size_values = [16]
bitsize_values = [32]
l1_reg_values = [1e-08]
l2_reg_values = [1e-08]


#List of Grid search parameters to iterate over
# learning_rate_values = [ 0.1, 0.01]
# alphas = [1e-03,  1e-5]
# margin_values = [24, 36, 48] 
# batch_size_values = [8, 24, 40, 48, 56]#16, 32, 64]
# bitsize_values = [32, 48]
# l2_reg_values = [0, 1e-08]
# l1_reg_values = [0, 1e-08]

#%%

for epochs in epoch_values:
    for learning_rate in learning_rate_values:
        for alpha in alphas:
            for margin in margin_values:
                for batch_size in batch_size_values:
                    for bitsize in bitsize_values:
                        for l1_reg in l1_reg_values:
                            for l2_reg in l2_reg_values:                         
                                # Train Valid & Test data
                                
                                
                                train_df, valid_test_df = get_data(data_path)
                                _, valid_prev = get_data(data_path_valid)
                                _, test_prev = get_data(data_path_test)
                                cols = list(train_df.columns[:10])
                                valid_test_df['id'] = range(valid_test_df.shape[0])
                                test_df = pd.merge(test_prev[cols], valid_test_df, on=cols)

                                diff_set = set(np.array(valid_test_df.id)) - set(np.array(test_df.id))
                                valid_df = valid_test_df[valid_test_df['id'].isin(diff_set)]
                                valid_df.drop(columns=['id'], inplace=True)

                                test_df.drop(columns=['id'], inplace=True)
                                print(valid_df.label_code.value_counts())
                                print(valid_df.shape)
                                print(test_df.label_code.value_counts())
                                print(test_df.shape)

                                # Verify the data set sizes based on Table 1 of the paper. 
                                print('Train data shape: ', train_df.shape)
                                print('Valid data shape: ', valid_df.shape)                        
                                print('Test data shape: ', test_df.shape)

                                similar_test_cols = [f'descrip_{i}' for i in range(1, 10) if (np.array(test_df[f'descrip_{i}']) == np.array(test_prev[f'descrip_{i}'])).all()]
                                similar_valid_cols = [f'descrip_{i}' for i in range(1, 10) if (np.array(valid_df[f'descrip_{i}']) == np.array(valid_prev[f'descrip_{i}'])).all()]

                                if len(similar_test_cols) != len(similar_valid_cols):
                                    raise ValueError("Lengths of similar columns don't match")

                                print("Sanity check passed!")

                                # Rename label_name column:   
                                train_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
                                valid_df.rename(columns = {'label_name': 'label_code'}, inplace = True)
                                test_df.rename(columns = {'label_name': 'label_code'}, inplace = True)

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

                                
                                valid_dataset_eval = CosfireDataset(valid_df)
                                valid_dataloader_eval = DataLoader(valid_dataset_eval, batch_size=batch_size, shuffle=False)
                                
                                

                                train_dataset_eval = CosfireDataset(train_df)
                                train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=batch_size, shuffle=False)
                                

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
                                
                                #####################################################################
                                #####################################################################

                                thresholds_abs_values = np.arange(-1, 1.2, 0.1)
                                mAP_results_valid = []
                                mAP_results_test = []
                                for _,thresh in enumerate(thresholds_abs_values):
                                    
                                    mAP_valid_thresh,_, _, _, _ = mAP_values(train_df,valid_df,thresh = thresh, percentile = False)
                                    mAP_test_thresh,_, _, _, _ = mAP_values(train_df, test_df,thresh = thresh, percentile = False)

                                    mAP_results_valid.append(mAP_valid_thresh)
                                    mAP_results_test.append(mAP_test_thresh)

                                # Plotting
                                plt.figure(figsize=(8, 6))
                                plt.plot(thresholds_abs_values, mAP_results_valid, label='mAP_valid')
                                plt.plot(thresholds_abs_values, mAP_results_test, label='mAP_test')
                                plt.xlabel('Threshold')
                                plt.ylabel('mAP')
                                plt.legend()
                                plt.savefig(output_dir + '/Maps_curves_abs_values.png')
                                plt.close()

                                data_abs_values = {'mAP_valid': mAP_results_valid,
                                        'mAP_test': mAP_results_test,
                                        'threshold': thresholds_abs_values}

                                df_thresh_abs_values = pd.DataFrame(data_abs_values)
                                df_thresh_abs_values.to_csv(output_dir +'/results_data_abs_values.csv',index = False)

                                # Find the index of the maximum mAP value
                                max_map_index = df_thresh_abs_values['mAP_valid'].idxmax()

                                # Retrieve the threshold corresponding to the maximum mAP
                                threshold_max_map_abs_values = df_thresh_abs_values.loc[max_map_index, 'threshold']
                                mAP_valid_abs_values,_, _, _, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map_abs_values, percentile = False)
                                mAP_test_abs_values,_, _, _, _ = mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False)


                                thresholds = range(0, 110, 10)

                                
                                mAP_results_valid = []
                                mAP_results_test = []
                                for _,thresh in enumerate(thresholds):
                                    
                                    maP_valid,_, _, _, _ = mAP_values(train_df,valid_df,thresh = thresh, percentile = True)
                                    mAP_test,_, _, _, _ = mAP_values(train_df, test_df,thresh = thresh, percentile = True)

                                    mAP_results_valid.append(maP_valid)
                                    mAP_results_test.append(mAP_test)
                                


                                data = {'mAP_valid': mAP_results_valid,
                                        'mAP_test': mAP_results_test,
                                        'threshold': thresholds}

                                df = pd.DataFrame(data)
                                df.to_csv(output_dir +'/results.csv',index = False)

                            
                                
                                # Find the index of the maximum mAP value
                                max_map_index = df['mAP_valid'].idxmax()

                                # Retrieve the threshold corresponding to the maximum mAP
                                threshold_max_map = df.loc[max_map_index, 'threshold']
                                mAP_valid1,_, _, _, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = True)
                                mAP_test1,_, _, _, _ = mAP_values(train_df, test_df,thresh = threshold_max_map, percentile = True)
                                
                                

                                #####################################################################
                                #####################################################################

                                
                                threshold_max_map = 0
                                maP_valid_zero,_, _, _, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = False)
                            
                                                    
                                mAP_test_zero,_, _, _, _ = mAP_values(train_df, test_df,thresh = threshold_max_map, percentile = False)

                                # Plotting
                                plt.figure(figsize=(8, 6))
                                plt.plot(thresholds, mAP_results_valid, label='mAP_valid')
                                plt.plot(thresholds, mAP_results_test, label='mAP_test')
                                plt.xlabel('Threshold')
                                plt.ylabel('mAP')
                                plt.legend()
                                #plt.axhline(y = maP_valid_zero, color = 'green') 
                                #plt.axhline(y = mAP_test_zero, color = 'r') 
                                plt.savefig(output_dir + '/Maps_curves.png')
                                plt.close()

                                # Create a figure and subplots
                                fig, axes = plt.subplots(1, 2, figsize=(13, 5))

                                # Plot 1: KDE curves for each label_code
                                for label_code in range(4):
                                    dff = test_df.query(f'label_code == {label_code}')
                                    out_array_test = []
                                    dd_test = np.array([out_array_test.extend(np.array(out)) for _, out in enumerate(dff.predictions)])
                                    out_array_test = np.array(out_array_test)
                                    sns.kdeplot(out_array_test, label=f'{dic_labels_rev[label_code]}', ax=axes[0])

                                # Customize plot 1
                                axes[0].set_xlim(-1, 1)
                                axes[0].set_xlabel('Value')
                                axes[0].set_ylabel('Density')
                                axes[0].legend(title='Galaxy Types')
                                axes[0].set_title('KDE Curves for Each Galaxy')


                                # Plot 2: KDE curve for all predictions
                                out_array_test = []
                                dd_test = np.array([out_array_test.extend(np.array(out)) for _, out in enumerate(test_df.predictions)])
                                out_array_test = np.array(out_array_test)
                                sns.kdeplot(out_array_test, color='r', ax=axes[1])

                                # Customize plot 2
                                axes[1].set_xlim(-1, 1)
                                axes[1].set_xlabel('Value')
                                axes[1].set_ylabel('Density')
                                axes[1].set_title('KDE Curves for All Galaxies')

                                # Adjust the spacing between subplots
                                plt.tight_layout()
                                plt.savefig(output_dir + '/Density_plot.png')
                                plt.close()

                                ################################################################

                                
                                fig, axes = plt.subplots(1, 3, figsize=(17, 5))

                                # Plot 1: Training data
                                array_dat = []
                                for i in range(train_df['predictions'].shape[0]):
                                    array_dat.append(list(train_df['predictions'].iloc[i]))
                                array_dat = np.array(array_dat)
                                y = train_df.label_name
                                tsne = TSNE(n_components=2, random_state=42)
                                z = tsne.fit_transform(array_dat)
                                df = pd.DataFrame()
                                df["y"] = y
                                df["comp-1"] = z[:,0]
                                df["comp-2"] = z[:,1]
                                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[0]).set(title="Train: T-SNE projection")

                                # Plot 2: Validation data
                                array_dat = []
                                for i in range(valid_df['predictions'].shape[0]):
                                    array_dat.append(list(valid_df['predictions'].iloc[i]))
                                array_dat = np.array(array_dat)
                                y = valid_df.label_name
                                tsne = TSNE(n_components=2, random_state=42)
                                z = tsne.fit_transform(array_dat)
                                df = pd.DataFrame()
                                df["y"] = y
                                df["comp-1"] = z[:,0]
                                df["comp-2"] = z[:,1]
                                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[1]).set(title="Valid: T-SNE projection")

                                # Plot 3: Testing data
                                array_dat = []
                                for i in range(test_df['predictions'].shape[0]):
                                    array_dat.append(list(test_df['predictions'].iloc[i]))
                                array_dat = np.array(array_dat)
                                y = test_df.label_name
                                tsne = TSNE(n_components=2, random_state=42)
                                z = tsne.fit_transform(array_dat)
                                df = pd.DataFrame()
                                df["y"] = y
                                df["comp-1"] = z[:,0]
                                df["comp-2"] = z[:,1]
                                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 4), data=df, ax=axes[2]).set(title="Test: T-SNE projection")

                                plt.tight_layout()

                                plt.savefig(output_dir + '/T-SNE_projection.png')
                                plt.close()

                                # Create a dictionary with the variable names and their values
                                data = {'optimal threshold': [threshold_max_map,threshold_max_map_abs_values,0],
                                        'Best Validation mAP': [maP_valid,mAP_valid_abs_values,maP_valid_zero],
                                        'Test mAP': [mAP_test1,mAP_test_abs_values,mAP_test_zero]
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
                                    'threshold_max_map': threshold_max_map,
                                    'threshold_max_map_abs_values': threshold_max_map,
                                    'mAP_valid_zero': maP_valid_zero,
                                    'mAP_test_zero': mAP_test_zero,
                                    'mAP_valid': mAP_valid1,
                                    'mAP_test': mAP_test1,
                                    'mAP_valid_abs_values': mAP_valid_abs_values,
                                    'mAP_test_abs_values': mAP_test_abs_values,
                                    'model_type': 'model_v3_layers_v2',                              
                                    'data_path': data_path.split('/')[1]
                                }

                                results_df = pd.DataFrame([results])
                                if not os.path.isfile(output_dir + "/model_selection_train_valid_and_test_13062024_v3_layers_v2.csv"):
                                    df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg', 'epochs', 'threshold_max_map', 'threshold_max_map_abs_values', 'mAP_valid_zero', 'mAP_test_zero', 'mAP_valid', 'mAP_test', 'mAP_valid_abs_values', 'mAP_test_abs_values', 'model_type', 'data_path'])
                                    results_df = pd.concat([df, results_df])
                                    results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_13062024_v3_layers_v2.csv", index=False)
                                else:
                                    df = pd.read_csv(output_dir + "/model_selection_train_valid_and_test_13062024_v3_layers_v2.csv")
                                    results_df = pd.concat([df, results_df], ignore_index=True)
                                    results_df.to_csv(output_dir + "/model_selection_train_valid_and_test_13062024_v3_layers_v2.csv", index=False)




# %%



