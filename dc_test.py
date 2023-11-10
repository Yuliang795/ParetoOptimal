from functions import * 
from sklearn.preprocessing import MinMaxScaler
import sys, os

# np.random.seed(1732)
epsilon_input=0.1
data_file_name = 'ionosphere'
data_file_path = "content/instance_" + data_file_name

# get the data frame and the specifications of the df
X,y,label_pos, num_rows, num_features, num_labels = get_df_and_spec(data_file_path)
# normalization on X
df = pd.DataFrame(MinMaxScaler().fit_transform(X) * 100,
                  index=X.index,
                  columns=X.columns)




## Distance Class
dc_time_counter_start = time.perf_counter()
## Distance Class
# Get distance class if drop the first DC, [1:]
pair_index_list = np.array(list(combinations(np.arange(df.shape[0]), 2)))
# pair_dist_df = pd.DataFrame()
# pair_dist_df['index'] = np.arange(len(pair_index_list))
# pair_dist_df['distance'] = np.linalg.norm(
# np.array(df.iloc[pair_index_list[:, 0]]) - np.array(df.iloc[pair_index_list[:, 1]]), axis=1)

epsilon=0.1
df_np = np.array(df)


pair_dist_array = np.linalg.norm(
df_np[pair_index_list[:, 0]] - df_np[pair_index_list[:, 1]], axis=1)
pair_dist_array_index_asce = np.argsort(pair_dist_array)

pair_index_list = pair_index_list.tolist()
# DC_index_list = [[]]
DC_Pair_list = [[]]
curr_least_value = pair_dist_array[pair_dist_array_index_asce[0]]
w_ind = 0
for i in pair_dist_array_index_asce:
    if abs(curr_least_value - pair_dist_array[i]) < epsilon:
        # DC_index_list[w_ind]+=[i]
        DC_Pair_list[w_ind]+=[pair_index_list[i]]

    else:
        # DC_index_list.append([i])
        DC_Pair_list.append([pair_index_list[i]])
        w_ind+=1
        curr_least_value = pair_dist_array[i]

# for i in DC_Pair_list[:10]:
#     print(i)
# print()
# sorted_pair_df = pair_dist_df.sort_values(by=['distance']).reset_index(drop=True)
# # (2) get distance class based on the sorted pair df using gready method
# pair_index_list = pair_index_list.tolist()
# DC_index_list = [[]]
# DC_Pair_list = [[]]
# curr_least_value = sorted_pair_df['distance'][0]
# w_ind = 0
# for i in range(sorted_pair_df.shape[0]):
#     if abs(curr_least_value - sorted_pair_df['distance'][i]) < epsilon:
#         DC_index_list[w_ind]+=[sorted_pair_df['index'][i]]
#         DC_Pair_list[w_ind]+=[pair_index_list[sorted_pair_df['index'][i]]]

#     else:
#         DC_index_list.append([sorted_pair_df['index'][i]])
#         DC_Pair_list.append([pair_index_list[sorted_pair_df['index'][i]]])
#         w_ind+=1
#         curr_least_value = sorted_pair_df['distance'][i]


dc_time_counter_end = time.perf_counter()

print(f'df: {df.shape}|dc:{dc_time_counter_end-dc_time_counter_start:.5f}| dc res:{len(DC_Pair_list)}')


ctrl_time_counter_start = time.perf_counter()
Distance_Class = get_distance_class(df, epsilon_input)
ctrl_time_counter_end = time.perf_counter()

print(f'ctrll | df: {df.shape}|dc:{ctrl_time_counter_end-ctrl_time_counter_start:.5f}| dc res:{len(Distance_Class)}')

def get_dist(a,b):
    return np.linalg.norm(df_np[a] - df_np[b])

for i in range(len(Distance_Class)):
    ctrl = np.array(Distance_Class[i]).reshape(-1,2)
    expr = np.array(DC_Pair_list[i]).reshape(-1,2)
    equal_by_row = np.all(ctrl == expr, axis=1)
    if not equal_by_row.all():
        diff_idx = np.where(equal_by_row == False)[0]
        for k in diff_idx:
            print(f'k: {k}')
            print(f'{ctrl[k]} - {expr[k]}  -- dist {get_dist(*ctrl[k]):.5f} - {get_dist(*ctrl[k]):.5f} | {True if get_dist(*ctrl[k]) == get_dist(*ctrl[k]) else False}')
            if get_dist(*ctrl[k]) != get_dist(*ctrl[k]):
                print(k)
                sys.exit()
        # print('\n')