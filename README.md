# Static geographical information of CityBike‘s stations in Brisbane 
This is an clustering analytics application to perform a Kmeans clustering based on the location of the bike stations. 
This application is build in the maven framework and the technlogies used are as follows:
```ruby
Scala 2.11
Spark 2.3
Spark Mllib 2.3
```
This application is deployed as a release version 1.0.
# Under the hood
Preparing Input Data
K-means algorithm works only on numerical features and so it needs to quantify the features in order to calculate distance between data objects. Here, we are implementing the KMeans on the locations (latitude and longitude)
```ruby
val df = spark.read.json(args(2))
val df2 = df.select("latitude", "longitude")

val assembler = new VectorAssembler().setInputCols(Array("latitude", "longitude")).setOutputCol("features")
val df3 = assembler.transform(df2).select("features")
```

Setting the Parameters: 
We need to set the following parameters to run K means algorithm which is taken as a program arguments:

```ruby
val numberOfClusters = 3
val numberOfSeeds = 1
```
Vizualizing Results: 
model.train returns KMeansModel object by which we can get the following information about the clusters of the datasets.
We can get centres of the clusters as:

```ruby
18/03/25 21:10:57 INFO Rules: Cluster Centres: 
18/03/25 21:10:57 INFO Rules: [-27.471353037037026,153.02793525925927]
18/03/25 21:10:57 INFO Rules: [-27.4886793,153.00309355000002]
18/03/25 21:10:57 INFO Rules: [-27.47459091176471,153.0124179117647]
18/03/25 21:10:57 INFO Rules: [-27.460040024390246,153.04255573170732]
```
We can get the prediction made on the model as:

```ruby
18/03/25 21:10:56 INFO Rules: Predictions: 
18/03/25 21:10:56 INFO Rules: [[-27.482279,153.028723],0]
18/03/25 21:10:56 INFO Rules: [[-27.47059,153.036046],0]
18/03/25 21:10:56 INFO Rules: [[-27.474531,153.042728],3]
18/03/25 21:10:56 INFO Rules: [[-27.461881,153.046986],3]
18/03/25 21:10:56 INFO Rules: [[-27.469658,153.016696],2]
18/03/25 21:10:56 INFO Rules: [[-27.48172,153.00436],1]
18/03/25 21:10:56 INFO Rules: [[-27.493626,153.001482],1]
18/03/25 21:10:56 INFO Rules: [[-27.476076,153.002459],2]
18/03/25 21:10:56 INFO Rules: [[-27.493963,153.011938],1]
18/03/25 21:10:56 INFO Rules: [[-27.482197,153.020894],2]
18/03/25 21:10:56 INFO Rules: [[-27.465226,153.050864],3]
18/03/25 21:10:56 INFO Rules: [[-27.468447,153.024662],0]
18/03/25 21:10:56 INFO Rules: [[-27.473021,153.025988],0]
18/03/25 21:10:56 INFO Rules: [[-27.457825,153.036866],3]
```

We can also get the silhouette with squared euclidean distance and WSSSE error score as follows:

```ruby
18/03/25 22:12:12 INFO Rules: Silhouette with squared euclidean distance = 0.588169797378973
18/03/25 22:12:12 INFO Rules: Within Set Sum of Squared Errors = 0.01379746828247333
```
# Launching Application with spark-submit
Once the application is bundled, it can be launched using the bin/spark-submit script. This script takes care of setting up the classpath with Spark and its dependencies, and can support different cluster managers and deploy modes that Spark supports.
The program arguments takes number of cluster(s), number of seed(s) and the resource file path and write results file path respectively. 

```ruby
./bin/spark-submit \
  --class main \
  --master local[*] \
  --executor-memory 20G \
  --deploy-mode cluster \
  /path/to/clustering-1.0.jar
  3 \
  1 \
  /path/to/resource/file/BrisbaneCityBike.json
  /path/to/write/results/results.txt
```
