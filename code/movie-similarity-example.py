from pyspark import SparkConf, SparkContext
from math import sqrt
import sys


#Boiler Plate code
conf = SparkConf().setMaster("local[*]").setAppName("item-item collaborative filtering")
sc = SparkContext(conf = conf)

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.item", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames
    

def filterDuplicates(userRatings):
    results = userRatings[1]
    (movie1,rating1) = results[0]
    (movie2,rating2) = results[1]
    return movie1 < movie2
    
def makePair(movieRatingPair):
    result1 = movieRatingPair[1][0]
    result2 = movieRatingPair[1][1]
    return ((result1[0],result2[0]),(result1[1],result2[1]))
    
def computeCosineSimilarity(ratingPair):
    numPair = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPair:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPair +=1
        
        num = sum_xy
        den = sqrt(sum_xx)*sqrt(sum_yy)
        
        score = 0
        if(den):
            score = num/float(den)
    return(score,numPair)

namedDict = loadMovieNames()
    
      
data = sc.textFile("file:///C:/SparkCourse/ml-100k/u.data")

ratings = data.map(lambda l: l.split()).map(lambda l:(int(l[0]),(int(l[1]),float(l[2]))))

joinedRatings = ratings.join(ratings)

#similarities between unique pairs of movies. convert (movie1,rating1)&(movie2,rating2) = (movie1,movie2)&(rating1,rating2)
moviePairs = joinedRatings.map(makePair)

# now we have to group the ratings for wach movie pair. convert to (movie1,movie2),((rating1,rating2),(rating1,rating2),(rating1,rating2)....)
moviePairGroupedRatings = moviePairs.groupByKey()

#compute cosine similarity
moviePairSimilarities = moviePairGroupedRatings.mapValues(computeCosineSimilarity).cache()


if(len(sys.argv) >1):
    scoreThreshold = 0.95
    coocuranceNumber = 1
    
    movieID = int(sys.argv[1])
    
    filteredResult = moviePairSimilarities.filter(lambda moviePair: \
    (moviePair[0][0] == movieID or moviePair[0][1] == movieID) and  \
    (moviePair[1][0] > scoreThreshold and moviePair[1][1] > coocuranceNumber))
    
    # Sort by quality score.
    results = filteredResult.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(5)
    
    print("Top 5 movies similar to the movie: " + namedDict[movieID])
    
    for result in results:
        (score,movie) = result
        movieWatchedID = movie[0]
        if(movieWatchedID == movieID):
            movieRecommended = movie[1]
        print(namedDict[movieRecommended]+ "\t score" + str(score[0])+ "\t strength" + str(score[1]))
            
        
        
        
    

    
    






