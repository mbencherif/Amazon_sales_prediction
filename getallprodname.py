import pandas as pd
import os
import glob

#l = [[]for i in range(49)]
l = []
for filename in os.listdir("."):
	if filename.startswith("US"):
		item_id = filename[3:-4]
		print(item_id)
		temp_df = pd.read_csv(filename, header=None)
		temp_df = temp_df.iloc[2:]
		asin = temp_df.iloc[0][0]
		print(asin)
		prod_name = temp_df.iloc[0][1]
		print(prod_name)
		l.append(item_id)
		l.append(asin)
		l.append(prod_name)

print(l)
print(len(l))

df = pd.DataFrame(index=[i for i in range(49)])
for i in range(0, 49):
	df.set_value(i, "item_id", l[3*i])
	df.set_value(i, "asin", l[3*i+1])
	prod_name = str(l[3*i+2])
	prod_name = prod_name[:7] + prod_name[15:]
	df.set_value(i, "prod_name", prod_name)
print(df)
df.to_csv('prod_name_asin_itemid.csv', sep=",", index=False)