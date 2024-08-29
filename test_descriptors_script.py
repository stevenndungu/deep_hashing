
#%%
from cosfire_workflow_utils import *
#%%
num = 1
data_path = f"test/descriptor_set_{num}_train_valid_test.mat" # Path to the Train_valid_test.mat file
data_path_valid = f"test/descriptor_set_{num}_train_valid.mat" # Path to the Train_valid.mat file

train_df, valid_test_df = get_data(data_path)
train_dfff, valid_prev = get_data(data_path_valid)

# %%
cols = list(valid_test_df.columns[:10])
valid_test_df['id'] = range(valid_test_df.shape[0])
valid_df = pd.merge(valid_prev[cols], valid_test_df, on=cols)
diff_set = set(np.array(valid_test_df.id)) - set(np.array(valid_df.id))
test_df = valid_test_df[valid_test_df['id'].isin(diff_set)]
valid_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)
# %%

dic_labels = { 'Bent':2,
  'Compact':3,
    'FRI':0,
    'FRII':1
}



def sanity_check(test_df,train_df, valid_df):
   df_test = pd.DataFrame(test_df.label_code.value_counts())
   tt1 = df_test.loc[dic_labels['Bent']].label_code == 103
   tt2 = df_test.loc[dic_labels['Compact']].label_code == 100
   tt3 = df_test.loc[dic_labels['FRI']].label_code == 100
   tt4 = df_test.loc[dic_labels['FRII']].label_code == 101
   if tt1 and tt2 and tt3 and tt4:
      print(f'Test folder is great')
   else:
      raise Exception(f'Test folder is incomplete!!')

   df_train = pd.DataFrame(train_df.label_code.value_counts())
   tt1 = df_train.loc[dic_labels['Bent']].label_code == 305
   tt2 = df_train.loc[dic_labels['Compact']].label_code == 226
   tt3 = df_train.loc[dic_labels['FRI']].label_code == 215
   tt4 = df_train.loc[dic_labels['FRII']].label_code == 434

   if tt1 and tt2 and tt3 and tt4:
      print(f'Train folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   df_valid = pd.DataFrame(valid_df.label_code.value_counts()) 
   tt1 = df_valid.loc[dic_labels['Bent']].label_code == 100
   tt2 = df_valid.loc[dic_labels['Compact']].label_code == 80
   tt3 = df_valid.loc[dic_labels['FRI']].label_code == 74
   tt4 = df_valid.loc[dic_labels['FRII']].label_code == 144

   if tt1 and tt2 and tt3 and tt4:
      print(f'Valid folder  is great')
   else:
      raise Exception(f'Test folder  is incomplete!!')
   print('##################################################')
   print('\n')
   print('##################################################')

sanity_check(test_df,train_df, valid_df)
# %%

dat1 = pd.read_csv("test/model_selection_train_valid_and_test_13062024_v3_layers_prev.csv")
dat2 = pd.read_csv("test/model_selection_train_valid_and_test_13062024_v3_layers1.csv")
dat2['id'] = range(dat2.shape[0])
dat3 = pd.read_csv("test/model_selection_train_valid_and_test_13062024_v3_layers19.csv")
dat3['id'] = range(dat3.shape[0])

#%%
keys = ['output_size', 'learning_rate', 'batch_size', 'alpha', 'margin', 'l1_reg', 'l2_reg']
df = pd.merge(dat1, dat2, on=keys)
diff_set = set(np.array(dat2.id)) - set(np.array(df.id))
df_sub = dat2[dat2['id'].isin(diff_set)]
df_sub.drop(columns=['id'], inplace=True)
df_sub = df_sub[keys]
df_sub.to_csv("test/hyperparams.csv", index=False)

df = pd.merge(dat1, dat3, on=keys)
diff_set2 = set(np.array(dat3.id)) - set(np.array(df.id))
df_sub2 = dat3[dat3['id'].isin(diff_set2)]
df_sub2.drop(columns=['id'], inplace=True)
df_sub2 = df_sub2[keys]


# %%
hyperparams = pd.read_csv("test/hyperparams.csv")
# %%






#%%


# %%
