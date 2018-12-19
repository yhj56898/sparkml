package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, LBFGS, LogisticGradient}
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * @author hejin-Yu
  *
  *
  *         二分类，
  *         基于LBFGS（内存受限的拟牛顿法，可以运用在【非线性】【线性】问题）上的数据挖掘
  *
  **/
object Lbfgs_logistic {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("LBFGS-logistic").build
    Logger.getRootLogger.setLevel(Level.WARN)

    val file = LocalFile.file_root + "regression\\sample_libsvm_data.txt"

    val data = MLUtils.loadLibSVMFile(sc, file)
    val randomSplit = data.randomSplit(Array(0.7, 0.3))
    val (train, test) = (randomSplit(0), randomSplit(1))

    try {
      train.cache()

      val gradient = new LogisticGradient()
      val upter = new L1Updater
      val numCorrections: Int = 10
      val tol: Double = 0.00001 //收敛数
      val numIters: Int = 20
      val regPara: Double = 0.1

      val numFeatures: Int = train.first().features.size
      val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))


      val __train = train.map {
        case (lp: LabeledPoint) => {
          (lp.label, lp.features)
        }
      }

      val (weightsWithIntercept, loss) =LBFGS.runLBFGS(__train, gradient, upter,
        numCorrections, tol, numIters, regPara, initialWeightsWithIntercept)

      val model = new LogisticRegressionModel(
        Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
        weightsWithIntercept(weightsWithIntercept.size - 1))


      /*-----------------逻辑回归 With LBFGS -----------------------------------------------*/

    } finally {
      train.unpersist(true)
    }

    sc.stop()
  }

}
