package com.hj.mllib.claz

import com.hj.util.{Algorithm, Params, RegType, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils

/**
  * @author hejin-Yu
  *
  *         二分类器
  *         对SVM,LR（逻辑回归）做的封装
  *
  **/
object BinaryClassification {

  def main(args: Array[String]) {

    val sc: SparkContext = SparkBuilder.appName("BinaryClassification")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)

    val file = "file:///E:\\idea-workspace\\ml\\sparkml\\deploy\\file\\mllib\\input\\classification\\sample_libsvm_data.txt"

    val params = Params(
      file
    )

    val examples = MLUtils.loadLibSVMFile(sc, params.input)

    examples.cache()

    val arrayData = examples.randomSplit(Array(0.8, 0.2))
    val (train, test) = (arrayData(0), arrayData(1))


    val updater = params.regType match {
      case RegType.L1 => new L1Updater
      case RegType.L2 => new SquaredL2Updater
      case _ => new SimpleUpdater
    }

    val model = params.algorithm match {
      case Algorithm.LR => {
        val algorithm = new LogisticRegressionWithLBFGS()
        algorithm.optimizer
          .setUpdater(updater)
          .setRegParam(params.regPara)
          .setNumIterations(params.num_iters)

        algorithm.run(train).clearThreshold()
      }
      case Algorithm.SVM => {
        val algorithm = new SVMWithSGD()

        algorithm.optimizer
          .setUpdater(updater)
          .setStepSize(params.step_size)
          .setNumIterations(params.num_iters)
          .setRegParam(params.regPara)

        algorithm.run(train).clearThreshold()
      }
    }


    val prediction = model.predict(test.map(_.features))
    val preWithLabel =  prediction.zip(test.map(_.label))

    /*--------------二分类评估

    评价指标主要有accuracy，precision，recall，F-score，
    以及 ROC和AUC
    ------------------------------------------------------*/
    val metrics = new BinaryClassificationMetrics(preWithLabel)

    println("areaUnderPR:"+metrics.areaUnderPR())
    println("areaUnderROC:"+metrics.areaUnderROC())


    examples.unpersist(true)
    sc.stop
  }

}




