# install dependencies
import pandas as pd
import csv

# remove header and unnecessary data
lines = []
with open(r"images.txt", 'r+') as fp:
    # read an store all lines into list
    lines = fp.readlines()
    # move file pointer to the beginning of a file
    fp.seek(0)
    # truncate the file
    fp.truncate()

    lines = lines[4:]
    lines = lines[::2]

    fp.writelines(lines)

# convert text file to pandas dataframe and store as csv
new = [lines[x].split() for x in range(0,len(lines))]

# field names
fields = ['id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera', 'frame']

with open('/content/drive/My Drive/Colab Notebooks/CV/localisation/values2.csv', 'w') as f:
	
	# using csv.writer method from CSV package
	write = csv.writer(f)
	
	write.writerow(fields)
	write.writerows(new)