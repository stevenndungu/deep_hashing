#%%
from cosfire_workflow_utils import *

# Hyperparameters
parser = argparse.ArgumentParser(description='COSFIRENet Training and Evaluation')
parser.add_argument('--input_size', type=int, default=372, help='Input size of the Descriptors')
parser.add_argument('--num', type=int, default=1)
args = parser.parse_args()
input_size = args.input_size
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

epochs = 2000

hyperparams = pd.read_csv("optimal_params_72bit.csv")
#print('Batch size: ',hyperparams.head(3))





def run():
    for row in range(hyperparams.shape[0]):             
         # Train Valid & Test data
                                             
         train_df, valid_test_df = get_data(data_path,dic_labels)
         _, valid_prev = get_data(data_path_valid,dic_labels)
         _, test_prev = get_data(data_path_test,dic_labels)

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
         
         train_dataloader = DataLoader(train_dataset, batch_size=int(hyperparams.loc[row, 'batch_size']), shuffle=True)
         
         
         # DataLoader for validation set
         valid_dataset = CosfireDataset(valid_df)
         valid_dataloader = DataLoader(valid_dataset, batch_size=int(hyperparams.loc[row, 'batch_size']), shuffle=True)
         
                     
         model = CosfireNet(input_size=input_size, bitsize=hyperparams.loc[row, 'output_size'], l1_reg=hyperparams.loc[row, 'l1_reg'], l2_reg=hyperparams.loc[row, 'l2_reg'])
         

         optimizer = optim.RMSprop(model.parameters(), lr=hyperparams.loc[row, 'learning_rate'])
         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

         # Lists to store training and validation losses
         train_losses = []
         val_losses = []
         # Variables to keep track of the best model

         # Train the loop
         for _ in tqdm(range(epochs), desc='Training Progress', leave=True):
               model.train()
               total_train_loss = 0.0
               for _, (train_inputs, labels) in enumerate(train_dataloader):
                  optimizer.zero_grad()
                  train_outputs, reg_loss = model(train_inputs)
                  loss = DSHLoss(u = train_outputs, y=labels, alpha = hyperparams.loc[row, 'alpha'], margin = hyperparams.loc[row, 'margin']) + reg_loss
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
                     val_loss = DSHLoss(u = val_outputs, y=val_labels, alpha = hyperparams.loc[row, 'alpha'], margin = hyperparams.loc[row, 'margin']) + reg_loss
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
         valid_dataloader_eval = DataLoader(valid_dataset_eval, batch_size=int(hyperparams.loc[row, 'batch_size']), shuffle=False)


         train_dataset_eval = CosfireDataset(train_df)
         train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=int(hyperparams.loc[row, 'batch_size']), shuffle=False)

         test_dataset = CosfireDataset(test_df)
         test_dataloader = DataLoader(test_dataset, batch_size=int(hyperparams.loc[row, 'batch_size']), shuffle=False)

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
               
               mAP_valid_thresh,_, _, _, _,_,_ = mAP_values(train_df,valid_df,thresh = thresh, percentile = False)
               mAP_test_thresh,_, _, _, _,_,_ = mAP_values(train_df, test_df,thresh = thresh, percentile = False)

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
         mAP_valid_abs_values,_, _, _, _, _, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map_abs_values, percentile = False)
         mAP_test_abs_values,_, _, _, _, _, _ = mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False)


         thresholds = range(0, 110, 10)


         mAP_results_valid = []
         mAP_results_test = []
         for _,thresh in enumerate(thresholds):
               
               maP_valid,_, _, _, _, _, _ = mAP_values(train_df,valid_df,thresh = thresh, percentile = True)
               mAP_test,_, _, _, _, _, _ = mAP_values(train_df, test_df,thresh = thresh, percentile = True)

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
         mAP_valid1,_, _, _, _,_, _ = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = True)
         mAP_test1,_, _, _, _,_,_ = mAP_values(train_df, test_df,thresh = threshold_max_map, percentile = True)



         #####################################################################
         #####################################################################


         threshold_max_map = 0
         maP_valid_zero,_, _, _, _,_,_ = mAP_values(train_df,valid_df,thresh = threshold_max_map, percentile = False)

                              
         mAP_test_zero,_, _, _, _,_,_ = mAP_values(train_df, test_df,thresh = threshold_max_map, percentile = False)

         # Plotting
         plt.figure(figsize=(8, 6))
         plt.plot(thresholds, mAP_results_valid, label='mAP_valid')
         plt.plot(thresholds, mAP_results_test, label='mAP_test')
         plt.xlabel('Threshold')
         plt.ylabel('mAP')
         plt.legend()
         plt.axhline(y = maP_valid_zero, color = 'green') 
         plt.axhline(y = mAP_test_zero, color = 'r') 
         plt.savefig(output_dir + '/Maps_curves.png')
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
               'output_size': hyperparams.loc[row, 'output_size'],
               'learning_rate': hyperparams.loc[row, 'learning_rate'],
               'batch_size': int(hyperparams.loc[row, 'batch_size']),
               'alpha': hyperparams.loc[row, 'alpha'],                                        
               "margin": hyperparams.loc[row, 'margin'], 
               "l1_reg": hyperparams.loc[row, 'l1_reg'],
               "l2_reg": hyperparams.loc[row, 'l2_reg'],
               'epochs': epochs,
               'threshold_max_map': threshold_max_map,
               'threshold_max_map_abs_values': threshold_max_map,
               'mAP_valid_zero': maP_valid_zero,
               'mAP_test_zero': mAP_test_zero,
               'mAP_valid': mAP_valid1,
               'mAP_test': mAP_test1,
               'mAP_valid_abs_values': mAP_valid_abs_values,
               'mAP_test_abs_values': mAP_test_abs_values,
               'model_type': 'model_v3_layers',                              
               'data_path': data_path.split('/')[2]
         }

         results_df = pd.DataFrame([results])
         if not os.path.isfile(output_dir + "/Optimal_72bit_04092024_v3_layers.csv"):
               df = pd.DataFrame(columns=['input_size', 'output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg', 'epochs', 'threshold_max_map', 'threshold_max_map_abs_values', 'mAP_valid_zero', 'mAP_test_zero', 'mAP_valid', 'mAP_test', 'mAP_valid_abs_values', 'mAP_test_abs_values', 'model_type', 'data_path'])
               results_df = pd.concat([df, results_df])
               results_df.to_csv(output_dir + "/Optimal_72bit_04092024_v3_layers.csv", index=False)
         else:
               df = pd.read_csv(output_dir + "/Optimal_72bit_04092024_v3_layers.csv")
               results_df = pd.concat([df, results_df], ignore_index=True)
               results_df.to_csv(output_dir + "/Optimal_72bit_04092024_v3_layers.csv", index=False)


         mAP, cum_prec, cum_recall = MapWithPR_values(train_df,test_df,thresh = threshold_max_map_abs_values, percentile = False)

         num_dataset=train_df.shape[0]
         index_range = num_dataset // 100
         index = [i * 100 - 1 for i in range(1, index_range + 1)]
         max_index = max(index)
         overflow = num_dataset - index_range * 100
         index = index + [max_index + i for i in range(1, overflow + 1)]
         c_prec = cum_prec[index]
         c_recall = cum_recall[index]

         pr_data = {
         "index": index,
         "P": c_prec.tolist(),
         "R": c_recall.tolist()
         }

         with open(output_dir + f'/COSFIRE_{row}.json', 'w') as f:
            f.write(json.dumps(pr_data))
         pr_data = {
            "COSFIRE": output_dir + f'/COSFIRE_{row}.json',
            "DenseNet": '/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/DenseNet.json'
         }
         N = 200
         # N = -1
         for key in pr_data:

            path = pr_data[key]
            pr_data[key] = json.load(open(path))


         # markers = "DdsPvo*xH1234h"
         markers = ".........................."
         method2marker = {}
         i = 0
         for method in pr_data:
            method2marker[method] = markers[i]
            i += 1

         plt.figure(figsize=(15, 5))
         plt.subplot(131)

         for method in pr_data:
            P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
            print(len(P))
            print(len(R))
            plt.plot(R, P, linestyle="-",  label=method)
         plt.grid(False)
         plt.xlim(0, 1)
         plt.ylim(0, 1)
         plt.xlabel('recall')
         plt.ylabel('precision')
         plt.legend()
         plt.subplot(132)
         for method in pr_data:
            P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
            plt.plot(draw_range, R, linestyle="-",  label=method)
         plt.xlim(0, max(draw_range))
         plt.grid(False)
         plt.xlabel('The number of retrieved samples')
         plt.ylabel('recall')
         plt.legend()

         plt.subplot(133)
         for method in pr_data:
            P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
            plt.plot(draw_range, P, linestyle="-",  label=method)
         plt.xlim(0, max(draw_range))
         plt.grid(False)
         plt.xlabel('The number of retrieved samples')
         plt.ylabel('precision')
         plt.legend()
         plt.savefig(output_dir + f"/pr_{row}.png")
         #plt.savefig(output_dir + f"/pr_{row}.svg")
         plt.close()


         label = 'mean_std'
         train_df_dn = pd.read_csv('/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_training.csv')
         test_df_dn = pd.read_csv('/home4/p307791/habrok/projects/deep_hashing_github/data_complete_Tue_Aug_13_20_43_02_2024/df_testing.csv')

         predictions_test = []
         predictions_train = []

         for i in range(train_df_dn.shape[0]):
            predictions_train.append(np.array(ast.literal_eval(train_df_dn.predictions[i])))

         for i in range(test_df_dn.shape[0]):
            predictions_test.append(np.array(ast.literal_eval(test_df_dn.predictions[i])))

         train_df_dn['predictions'] = predictions_train
         test_df_dn['predictions'] = predictions_test

         threshold_max_map_abs_values = -0.8
         topk=100
         mAP_dn,mAP_std_dn,mAP_values_dn, r_binary_dn, train_label_dn, q_binary_dn, valid_label_dn = mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
         print('mAP top 100  for DenseNet is: ',np.round(mAP_dn,2))

         topk_number_images_dn = list(range(10,205,5))
         mAP_topk_dn = []
         mAP_topk_std_dn = []
         map_values_list = []
         for _, topk in enumerate(topk_number_images_dn):
            maP_dn,mAP_std_dn,map_values, _, _, _, _= mAP_values(train_df_dn, test_df_dn,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
            mAP_topk_dn.append(maP_dn)
            mAP_topk_std_dn.append(mAP_std_dn)
            map_values_list.append(map_values)
            

         data_densenet = {'topk_number_images': topk_number_images_dn,
                'mAP': mAP_topk_dn,
                'mAP_std': mAP_topk_std_dn}
         df_densenet = pd.DataFrame(data_densenet)
         df_densenet.to_csv(output_dir + f'/mAP_vs_{topk}_images_72bit_densenet_{label}_{row}.csv', index = False)


         topk_number_images = list(range(10,205,5))
         mAP_topk = []
         mAP_topk_std = []
         map_values_list = []
         for _, topk in enumerate(topk_number_images):
            maP,mAP_std,map_values, _, _, _,_= mAP_values(train_df, test_df,thresh = threshold_max_map_abs_values, percentile = False,topk=topk)
            mAP_topk.append(maP)
            mAP_topk_std.append(mAP_std)
            map_values_list.append(map_values)

         data = {'topk_number_images': topk_number_images,
                'mAP': mAP_topk,
                'mAP_std': mAP_topk_std}
         df_cosfire = pd.DataFrame(data)
         df_cosfire.to_csv(output_dir + f'/mAP_vs_topk_images_72bit_cosfire_{label}_{row}.csv', index = False)


         #fig, ax = plt.subplots(figsize=(10/3, 3))
         sns.lineplot(data=df_cosfire, x='topk_number_images', y='mAP',label='Cosfire',  color='red')
         # plt.fill_between(df_cosfire['topk_number_images'], df_cosfire['mAP'] - df_cosfire['mAP_std'], df_cosfire['mAP'] + df_cosfire['mAP_std'], color='#439CEF')
         sns.lineplot(data=df_densenet, x='topk_number_images', y='mAP',label='DenseNet',  color='blue')
         # plt.fill_between(df_densenet['topk_number_images'], df_densenet['mAP'] - df_densenet['mAP_std'], df_densenet['mAP'] + df_densenet['mAP_std'], color='#ffbaba')

         plt.xlabel('The number of retrieved samples')
         plt.ylabel('mAP')
         #plt.ylim(80, 92)
         plt.savefig(output_dir + f'/map_vs_topk_number_images_densenet_{label}_cbd_{row}.png')
         plt.savefig(output_dir + f'/map_vs_topk_number_images_densenet_{label}_cbd_{row}.svg',format='svg', dpi=1200)
         plt.close()




        

if __name__ == '__main__' :
    run()
