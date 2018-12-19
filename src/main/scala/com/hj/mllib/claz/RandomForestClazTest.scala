package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

/**
  * @author hejin-Yu
  *
  *
  **/
object RandomForestClazTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("随机森林-分类").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    val file =LocalFile.file_root+"regression\\sample_libsvm_data.txt"
    val data= MLUtils.loadLibSVMFile(sc,file)

    val splits =data.randomSplit(Array(0.8,0.2))
    val (train,test) =(splits(0),splits(1))

    val numClaz: Int = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees: Int = 3 /*实际中，应当使用更多*/

    /*----------
    Number of features to consider for splits at each node.
                                 Supported: "auto", "all", "sqrt", "log2", "onethird".
                               If "auto" is set, this parameter is set based on numTrees:
                                   if numTrees == 1, set to "all";
                                   if numTrees > 1 (forest) set to "sqrt".


    ----------------------------------------------------*/
    val featureSubsetStrategy: String = "auto"
    val impurity = "gini" //"gini" (recommended) or "entropy".

    val maxDepth:Int = 5
    val maxBins:Int = 32

    try{
      train.cache()
      val mod = RandomForest.trainClassifier(train,
        numClaz,categoricalFeaturesInfo,
        numTrees,featureSubsetStrategy,
        impurity,maxDepth,maxBins
      )

      val pre_label = test.map(x=>{
        (mod.predict(x.features),x.label)
      })

      val MSE = pre_label.map(xx=>{
        math.pow(xx._1 - xx._2 ,2)
      }).mean()

      val RMSE =math.sqrt(MSE)

      println(s"基于随机森林，RMSE=${RMSE}")

    }finally {
      train.unpersist(true)
    }

    sc.stop()
  }

}
