# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv

class Point:
    def __init__(self,essay_id,essay_set,essay_str,r1d1,r2d1):
        this.essay_id = essay_id
        this.essay_set = essay_set
        this.essay_str = essay_str
        this.r1d1 = r1d1
        this.r2d1 = r2d1
        this.features = []
    def __str__():
        feature_str = ','.join(this.features)
        return ','.join([this.essay_id,this.essay_set,str(this.r2d1 + this.r1d1),feature_str])

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
