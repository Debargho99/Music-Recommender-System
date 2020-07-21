# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:12:19 2020

@author: Debargho Basak
"""

from pyspark.mllib.recommendation import *
import random
from operator import *
import itertools

#loading data
lines = sc.textFile("artist_data_small.txt")
artistData = lines.map(lambda x: x.split("\t")).map(lambda x: (int (x[0]), x[1]))

lines = sc.textFile("artist_alias_small.txt")
artistAlias = lines.map(lambda x: x.split("\t")).map(lambda x: (int (x[0]), int(x[1])))
alias = artistAlias.collect()

lines = sc.textFile("user_artist_data_small.txt")
def keyVal(x):
    part = x.split(" ")
    for x1, x2 in alias:
        if int(part[1]) == x1:
            return (int(part[0]), x2, int(part[2]))
        else:
            return (int(part[0]), int(part[1]), int(part[2]))
userArtistData = lines.map(keyVal)



total_play_count = sc.parallelize(userArtistData.map(lambda x: (x[0], x[2])).reduceByKey(add).takeOrdered(3, lambda x: -x[1]))
total_artist_count = userArtistData.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], len(list(x[1]))))
final_join = total_play_count.join(total_artist_count).collect()

final_join = sorted(final_join, key=lambda x: x[1][0], reverse=True)
for item in final_join:
    print "User %d has a total play count of %d and a mean play count of %d" \
        %(item[0], item[1][0], item[1][0]/item[1][1])
        
        
trainData, validData, testData = userArtistData.randomSplit([4, 4, 2], seed=13)
print trainData.take(3)
print validData.take(3)
print testData.take(3)
print trainData.count()
print validData.count()
print testData.count()

trainData.cache()
testData.cache()
validData.cache()



def modelEval(model, dataset):
    userList = dataset.map(lambda x: x[0]).collect()
    allArtist = userArtistData.map(lambda x: x[1]).collect()
    
    trainList = trainData.map(lambda x: (x[0], x[1])).groupByKey()\
            .map(lambda x: (x[0], list(x[1]))).collect()
    trainDict = dict((x[0], x[1]) for x in trainList)
    
    dataList = dataset.map(lambda x: (x[0], x[1])).groupByKey()\
            .map(lambda x: (x[0], list(x[1]))).collect()
    dataDict = dict((x[0], x[1]) for x in dataList)
    
    score = 0.0
    
    for user in userList:
        nonTrainArtist = set(allArtist) - set(trainDict[user])
        artist = map(lambda x: (user, x), nonTrainArtist)
        artist = sc.parallelize(artist)
        
        length = len(dataDict[user])

        prediction = model.predictAll(artist)
        
        predictionList = prediction.map(lambda x: (x.product, x.rating))\
            .takeOrdered(length, key=lambda x: -x[1])
        predictionList = sc.parallelize(predictionList)\
            .map(lambda x: x[0]).collect()
            
        overlap = set(predictionList).intersection(dataDict[user])
        score += (len(overlap) / float(length))
        
    return (score/len(userList))

#training model
rank = [2, 10, 20]
for i in rank:
    model = ALS.trainImplicit(trainData, rank = i, seed=345)
    print "The model score for rank %d is %f" %(i, modelEval(model, validData))

#checking accuracy of model
bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData)


#results
recommendation = bestModel.recommendProducts(1059637, 5)
for index, reco in enumerate(recommendation):
    print "Artist %d: %s" %(index, str(artistData.lookup(key=reco.product)[0])) 


        