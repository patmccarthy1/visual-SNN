a = [1,2,3]
b = [4,5,6]

c = [a.insert(0,"index"),b.insert(0,"time")]

import csv

name = "test"

with open(name+".csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(c)

def spikes_to_csv(file_name,spike_array):
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(spike_array)
        
spikes_to_csv("test2",c)
