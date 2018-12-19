package com.hj.mllib.features

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *
  **/
object PCAtest2 {

  def main(args: Array[String]) {
    val file = LocalFile.file_root + "a.mat"

    val sc = SparkBuilder.appName("主成分析").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val data:RDD[LabeledPoint] = genData(sc, file)

    val pca =new PCA(3).fit(data.map(_.features))


    /*----------------通过PCA模型，重置特征------------------------------------------------*/

    val transDF:RDD[LabeledPoint] =data.map(lp=>{
      lp.copy(features = pca.transform(lp.features))
    })

    transDF.foreach(println(_))


    sc.stop()
  }

  def genData(sc: SparkContext, file: String): RDD[LabeledPoint] = {

    val data: RDD[LabeledPoint] = sc.textFile(file).map(t => {
      val arr = t.split(" ").map(_.toDouble)
      val lab: Double = if (arr(0) > 1) 1.0 else 0
      LabeledPoint(lab, Vectors.dense(arr))
    })

    data
  }

}
