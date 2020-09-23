import pandas as pd
import math

def myknn_eucledian(x_train,y_train,x_test,k):
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()
    x_test = x_test.values.tolist()
    result = []
    for i in range(len(x_test)):
        distance =[]
        for k1 in range(len(x_train)):
            dist = 0
            for j in range(len(x_train[k])):
                dist+=math.sqrt((x_test[i][j]-x_train[k1][j])*(x_test[i][j]-x_train[k1][j]))
                #dist+=abs(x_test[i][j]-x_train[k1][j])
            distance.append((dist,k1))
        distance.sort()
        distance = distance[:k]
        countzero = 0
        countone = 0
        for w in distance:
            if(y_train[w[1]] ==0):
                countzero+=1
            else:
                countone+=1
        if(countzero>countone):
            result.append(0)
        else:
            result.append(1)
    return pd.DataFrame(result)


def myknn_manhattan(x_train,y_train,x_test,k):
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()
    x_test = x_test.values.tolist()
    result = []
    for i in range(len(x_test)):
        distance =[]
        for k1 in range(len(x_train)):
            dist = 0
            for j in range(len(x_train[k])):
                #dist+=math.sqrt((x_test[i][j]-x_train[k1][j])*(x_test[i][j]-x_train[k1][j]))
                dist+=abs(x_test[i][j]-x_train[k1][j])
            distance.append((dist,k1))
        distance.sort()
        distance = distance[:k]
        countzero = 0
        countone = 0
        for w in distance:
            if(y_train[w[1]] ==0):
                countzero+=1
            else:
                countone+=1
        if(countzero>countone):
            result.append(0)
        else:
            result.append(1)
    return pd.DataFrame(result)
            
            
def myknn_minkowski(dist,x_train,y_train,x_test,k):
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()
    x_test = x_test.values.tolist()
    result = []
    import scipy.spatial 
    for i in range(len(x_test)):
        distance =[]
        for k1 in range(len(x_train)):
            dist = scipy.spatial.distance.minkowski(x_test[i],x_train[k1] ,1)
            #for j in range(len(x_train[k])):
                #dist+=math.sqrt((x_test[i][j]-x_train[k1][j])*(x_test[i][j]-x_train[k1][j]))
             #   dist+=abs(x_test[i][j]-x_train[k1][j])
            distance.append((dist,k1))
        distance.sort()
        distance = distance[:k]
        countzero = 0
        countone = 0
        for w in distance:
            if(y_train[w[1]] ==0):
                countzero+=1
            else:
                countone+=1
        if(countzero>countone):
            result.append(0)
        else:
            result.append(1)
    return pd.DataFrame(result)

def myknn_weighted(x_train,y_train,x_test,k,weights):
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()
    x_test = x_test.values.tolist()
    result = []
    for i in range(len(x_test)):
        distance =[]
        for k1 in range(len(x_train)):
            dist = 0
            for j in range(len(x_train[k1])):
                dist+=(x_test[i][j]-x_train[k1][j])*(x_test[i][j]-x_train[k1][j])*weights[j]
            distance.append((dist,k1))
        distance.sort()
        distance = distance[:k]
        countzero = 0
        countone = 0
        for w in distance:
            #print(w)
            #print(y_train[w[1]])
            #print("done")
            if(y_train[w[1]] == 0):
                countzero+=1
#                 print("########here#####")
            else:
                countone+=1
        if(countzero>countone):
            result.append(0)
            #print("########here#####")
        else:
            result.append(1)
    return pd.DataFrame(result)
            
            
            
     