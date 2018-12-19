package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils

/**
  * @author hejin-Yu
  *
  *
  *         上述的 标准差貌似有点大，初步为9.01263575415721
  *         env:
  *         L1Updater
  *         regParam =0.1
  *         numIters = 30
  *         stepSize =1.0
  *
  *         ---------
  *         env:
  *         *   L1Updater
  *         *   regParam =0.01
  *         *   numIters = 30
  *         *   stepSize =1.0
  *         ----------------------------->标准差为10.081983625781183
  *
  *         ---------
  *         env:
  *         *   L1Updater
  *         *   regParam =0.01
  *         *   numIters = 20
  *         *   stepSize =1.0
  *         * ----------------------------->标准差为10.135105996116891
  *
  *
  *         ---------
  *         env:
  *         *   L2Updater
  *         *   regParam =0.01
  *         *   numIters = 20
  *         *   stepSize =1.0
  *         * ----------------------------->标准差为10.734119948010907
  *
  *
  *         numIters = 100--->10.333875356678488
  *
  *         ..................总体而言，当前：线性回归预测，标准差值徘徊在(9.1 ~ 10.7)之间
  *
  **/
object LinearRegTest2 {

  def main(args: Array[String]) {


    val sc = SparkBuilder.appName("线性回归模型").build
    Logger.getRootLogger.setLevel(Level.WARN)


    val file = LocalFile.file_root + "regression\\sample_linear_regression_data.txt"

    val data = MLUtils.loadLibSVMFile(sc, file)
    val randomSplit = data.randomSplit(Array(0.8, 0.2))
    val (train, test) = (randomSplit(0), randomSplit(1))

    val upStr = "L2"
    val upter: Updater = {
      upStr match {
        case "L1" => new L1Updater
        case "L2" => new SquaredL2Updater
        case _ => new SimpleUpdater
      }
    }

    val regParam: Double = 0.01 //其他条件不变，试下正则化系数由0.1--> 0.01下的结果
    val numIters: Int = 100
    val stepSize: Double = 1.0


    val alg = new LinearRegressionWithSGD()
    alg.optimizer
      .setUpdater(upter)
      .setRegParam(regParam)
      .setNumIterations(numIters)
      .setStepSize(stepSize)

    val mod = alg.run(train)

    try {
      train.cache()

      val pre_lable = test.map {
        case (lp: LabeledPoint) => {
          (mod.predict(lp.features), lp.label)
        }
      }

      /*      val actual:Double =  pre_lable.filter{
              case (pre:Double,lab:Double)=>{
                (pre == lab)
              }
            }.count().toDouble / pre_lable.count()


            println(s"预测值与真实值，相一致下的准确性=${actual}") //回归问题，label不是简单的定义值

       */
      /*-------------计算标准差------------------------------------------------------*/
      val MSE: Double = pre_lable.map(t => {
        math.pow(t._1 - t._2, 2)
      }).mean()

      val RMSE = math.sqrt(MSE)
      println(s"回归问题下:由线性回归(With SGD)模型预测的标准差为${RMSE}")

    } finally {
      train.unpersist(true)
    }


    sc.stop()


  }

}
