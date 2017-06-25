# Taxi Trajectory Prediction

## Descriptions

https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i

## Data

https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

## Model

Experimented in: Word2vec, K-means, Mean Shift clustering, Deep Learning

* Using k-means algorithm for clustering destination points.
* Using the word2vec model to embed each point in the map, datetime, taxiId, ... to a real numberic vector.
* Using multi-layer perceptron (Deep Learning) to predict the destination of the taxi trip to one of clusters.
* Using reduce_mean of points in the cluster to predict final destiantion of the taxi trip.
