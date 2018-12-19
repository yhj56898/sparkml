package com.hj.mllib.cluster

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         KMeans
  *         ---> 十大经典数据挖掘算法之一
  *
  *         决策树 k-means   Svm
  *         挖掘布尔关联规则频繁项集（Apriori ）
  *
  *         PageRank
  *         kNN
  *         Naive Baye
  *
  **/
object KMeansTest {

  def main(args: Array[String]) {

    val file = LocalFile.file_root + "mllibFromSpark\\iris.data"

    val sc = SparkBuilder.appName("KMeans聚类：近朱者赤近墨者黑")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)

    val data: RDD[(String, Vector)] = sc.textFile(file).map { x => {
      val d = x.split(",")
      val label: String = d(4)
      val f = Vectors.dense(d.dropRight(1).map(_.toDouble))
      (label, f)
    }
    }

    data.cache()
    val randomSplit = data.randomSplit(Array(0.8, 0.2))
    val (train,test) =(randomSplit(0),randomSplit(1))

    val in_kmeans_initMode: String = "random"

    val initMode = in_kmeans_initMode match {
      case "random" => KMeans.RANDOM
      case "parallel" => KMeans.K_MEANS_PARALLEL
    }

    val k: Int = 3
    /*使用鸢尾花数据集，该数据集中有3个样本类*/
    val numIters: Int = 20

    /**
      * Set the initialization algorithm. This can be either "random" to choose random points as
      * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
      * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
      */
    val model = new KMeans()
      .setInitializationMode(initMode)
      .setK(k)
      .setMaxIterations(numIters)
    //.setEpsilon() //收敛值
      .run(train.map(_._2))

    val pre_label:RDD[(String,Int)] =test.map{
      case (label:String,fs:Vector)=>{
        ( label,model.predict(fs) )
      }
    }

    /*---------------------打印下，原始的标签 与预测出的 K -----------------------------------------*/
    pre_label.foreach(it=>{
      println(s"label:${it._1}\t k:${it._2}")
    })

    data.unpersist(true)
    sc.stop
  }

}
