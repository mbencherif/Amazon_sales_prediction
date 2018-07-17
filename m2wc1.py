import numpy as np
import pandas as pd
import sys
from operator import itemgetter
from itertools import groupby
from random import randint

filename = sys.argv[1]

week_list = pd.date_range(start='01-01-2015', end='01-01-2018', freq='W')
print(week_list)

filedf = pd.read_csv(filename, header=None)
print(filedf)

new_df = pd.DataFrame(index = week_list, columns = ["sale_count","date_tup"])
index_tuple_list = [] 
for i in range(new_df.shape[0]):
	#print(str(i)[:10])
	dt = new_df.index.values[i]
	year = int(str(dt)[:4])
	month = int(str(dt)[5:7])
	day = int(str(dt)[8:10])
	temp_tup = (month, day, year)
	new_df.iloc[i]["date_tup"] = temp_tup
	index_tuple_list.append(temp_tup)

print(new_df)

print(index_tuple_list)

file_date_tup_list = []
file_val_list = []
for i in range(filedf.shape[0]):
	dateinfile = filedf.iloc[i][0]
	file_val_list.append(int(filedf.iloc[i][1]))
	#print(filedf.iloc[i][0])
	day = int(dateinfile[:2])
	month = int(dateinfile[3:5])
	year = int(dateinfile[6:])
	temp_tup = (month, day, year)
	file_date_tup_list.append(temp_tup)

fl_dt_tup_ser = pd.Series(file_date_tup_list)
filedf['dt_tup'] = fl_dt_tup_ser.values

print(filedf)
print(file_date_tup_list)
print(file_val_list)
week_to_mon_dict = {}
for grp, elmts in groupby(index_tuple_list, itemgetter(0, 2)):
    week_to_mon_dict[grp] = len(list(elmts))
    print(grp, len(list(elmts)))
print(week_to_mon_dict)
wkno = 4

i = 0
iLength = new_df.shape[0]
print (iLength)


while i < iLength:

	for j in range(filedf.shape[0]):
		if new_df.iloc[i]['date_tup'][0] == filedf.iloc[j]['dt_tup'][0] and new_df.iloc[i]['date_tup'][2] == filedf.iloc[j]['dt_tup'][2]:
			k = new_df.iloc[i]['date_tup'][0]
			v = new_df.iloc[i]['date_tup'][2]
			for kk,vv in week_to_mon_dict.items():
				if k==kk[0] and v==kk[1]:
					wkno = int(vv)
			randintlist = []
			threshold = int(filedf.iloc[j][1] /wkno)
			print(threshold)
			for m in range(wkno-1):
				randintlist.append(randint(int(threshold/2),threshold))
				new_df.iloc[i]['sale_count'] = randint(0,threshold)
			sumrandint = sum(randintlist)
			randintlist.append(int(filedf.iloc[j][1] - sumrandint))
			#print(randintlist)
			#new_df.iloc[i]['sale_count'] = filedf.iloc[j][1]/wkno
			#new_df.iloc[i]['sale_count'] = randintlist[randint(0, len(randintlist)-1)]
			print(randintlist)

			for n in range(wkno):
				new_df.iloc[i+n]['sale_count'] = randintlist[n]
			i = i + wkno


print(new_df)

ready_df = new_df.drop('date_tup',1)
print(ready_df)
print(ready_df.columns)

newfilename = filename[:4] + "-" +filename[4:6] + ".csv"
ready_df.to_csv(newfilename, sep=",")
