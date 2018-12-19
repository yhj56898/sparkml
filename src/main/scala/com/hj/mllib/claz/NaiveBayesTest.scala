package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         朴素贝叶斯
  *
  *         * Trains a Naive Bayes model given an RDD of `(label, features)` pairs.
  *         *
  *         * This is the default Multinomial NB ([[http://tinyurl.com/lsdw6p]]) which can handle all
  *         * kinds of discrete(非连续的、离散的) data.  For example, by converting documents into TF-IDF vectors, it
  *         * can be used for document classification.
  **/
object NaiveBayesTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("朴素贝叶斯分类").build
    val file = LocalFile.file_root + "mllibFromSpark\\sample_naive_bayes_data.txt"
    /*0,1 0 0*/
    val data: RDD[LabeledPoint] = sc.textFile(file).map(t => {
      val arr = t.split(",")
      val label = arr(0).toDouble
      val fs = arr(1).split(" ").map(_.toDouble)
      LabeledPoint(label, Vectors.dense(fs))
    })

    val splits = data.randomSplit(Array(0.8, 0.2))
    val (train, test) = (splits(0), splits(1))
    try {
      train.cache()

      val lambda: Double =1.0 // The smoothing parameter
      val modelType: String = "multinomial" // The type of NB model to fit from the enumeration NaiveBayesModels, can be
      //multinomial or bernoulli

      val alg = NaiveBayes.train(train,lambda,modelType)

      /*-----------------基于词频处理后，
      可以用来做 文档分类
      -------------------------------------------*/

      val pre_lable =test.map(lp=>{
        (alg.predict(lp.features),lp.label)
      })

      val MSE = pre_lable.map(t=>{
        math.pow( t._1 - t._2,2)
      }).mean()

      val RMSE = math.sqrt(MSE)

      println(s"当前数据集，基于朴素贝叶斯分类模型，RMSE =${RMSE}")

    } finally {
      train.unpersist(true)
    }


    sc.stop
  }

}
