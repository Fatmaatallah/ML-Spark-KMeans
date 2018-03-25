import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

object main {

  //Initialize logging
  val logger: Logger = Logger.getLogger("My.Example.Code.Rules")
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.apache.spark.storage.BlockManager").setLevel(Level.ERROR)
  logger.setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    //Creating a spark session
    val spark = SparkSession
      .builder()
      .appName("Clustering for Brisbane_CityBike")
      .master("local")
      .getOrCreate()

    //Set number of cluster(s) and seed(s)
    val numberOfClusters = args(0).toInt
    val numberOfSeeds = args(1).toLong

    // Trains a k-means model.
    val df = spark.read.json(args(2))
    val df2 = df.select("latitude", "longitude")

    val assembler = new VectorAssembler().setInputCols(Array("latitude", "longitude")).setOutputCol("features")
    val df3 = assembler.transform(df2).select("features")

    //Training model with KMeans
    logger.info("number of cluster(s): " + args(0))
    logger.info("number of seed(s): " + args(1))
    val kmeans = new KMeans().setK(numberOfClusters).setSeed(numberOfSeeds)
    val model = kmeans.fit(df3)

    // Make predictions
    val predictions = model.transform(df3)
    logger.info("Predictions: ")
    predictions.collect().foreach(logger.info(_))

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    logger.info(s"Silhouette with squared euclidean distance = $silhouette")

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(df3)
    logger.info(s"Within Set Sum of Squared Errors = $WSSSE")

    // Show the result
    logger.info("Cluster Centres: ")
    model.clusterCenters.foreach(logger.info(_))
  }
}