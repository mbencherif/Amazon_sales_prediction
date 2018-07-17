import pandas as pd
import os
import glob

ml = []
p70l = []
p80l = []
p90l = []
for i in range(1,27):
	mlc = "week"+str(i)+"mean"
	ml.append(mlc)
	p70lc = "week"+str(i)+"p70"
	p70l.append(p70lc)
	p80lc = "week"+str(i)+"p80"
	p80l.append(p80lc)
	p90lc = "week"+str(i)+"p90"
	p90l.append(p90lc)
main_col = ml + p70l + p80l + p90l
#print(main_col)
#print(len(main_col))

week_list = pd.date_range(start='06-24-2018', periods = 26, freq='W')
print(week_list)

def get_values_in_columns(sample_dataframe):
	sample_dataframe.index = [i for i in range(temp_df_trans.shape[0])]
	mean_val = pd.Series(sample_dataframe.iloc[:26][2]).tolist()
	p70_val = pd.Series(sample_dataframe.iloc[26:52][2]).tolist()
	p80_val = pd.Series(sample_dataframe.iloc[52:78][2]).tolist()
	p90_val = pd.Series(sample_dataframe.iloc[78:][2]).tolist()
	temp_new_df = pd.DataFrame([mean_val,p70_val,p80_val,p90_val])
	last_df_t = temp_new_df.transpose()
	last_df_t.columns = ["mean_val","p70_val","p80_val","p90_val"]
	last_df_t.index = week_list
	return last_df_t

for filename in os.listdir("."):
	if filename.startswith("US"):
		print(filename)

		temp_df = pd.read_csv(filename, header=None)
		temp_df = temp_df.iloc[2:]
		#print(temp_df.iloc[0][:])
		del_columns = [i for i in range(28)]
		temp_df.drop(temp_df.columns[del_columns], axis=1, inplace=True)
		temp_df.columns = main_col
		temp_df_trans = temp_df.T
		new_arranged_df = get_values_in_columns(temp_df_trans)
		newfilename = filename[3:-4] + "_pred.csv"
		new_arranged_df.to_csv(newfilename, sep= ',')

'''
print(temp_df)
print(temp_df_trans)

#temp_df_trans.columns = ["mean_val", "p70_val", "p80_val", "p90_val"]
temp_df_trans.index = [i for i in range(temp_df_trans.shape[0])]
#print(temp_df_trans)
mean_val = pd.Series(temp_df_trans.iloc[:26][2]).tolist()
print(mean_val)
p70_val = pd.Series(temp_df_trans.iloc[26:52][2]).tolist()
print(p70_val)
#temp_df_trans["p70_val"] = p70_val
p80_val = pd.Series(temp_df_trans.iloc[52:78][2]).tolist()
print(p80_val)
#temp_df_trans["p80_val"] = p80_val
p90_val = pd.Series(temp_df_trans.iloc[78:][2]).tolist()
print(p90_val)
#temp_df_trans["p90_val"] = p90_val
#print(temp_df_trans)

last_df = pd.DataFrame([mean_val,p70_val,p80_val,p90_val])
print(last_df)
last_df_t = last_df.transpose()
print(last_df_t)
last_df_t.columns = ["mean_val","p70_val","p80_val","p90_val"]
print(last_df_t)
'''