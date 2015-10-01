# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv

def make_points():
    with open('training_set.csv', 'rb') as f:
        csv_rows = list(csv.reader(f, delimiter = ','))
        i = 1
        out_file = open('training_'+str(i)+'.csv','w') 
        writer = csv.writer(out_file)
        for row in csv_rows:
            if row[1] == str(i):
                writer.writerows([row])
            else:
                out_file.close()
                i += 1
                out_file = open('training_'+str(i)+'.csv','w')
                writer = csv.writer(out_file)

if __name__=='__main__':
    make_points()
