package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.{Logger,Level}
/**
  * @author hejin-Yu
  *
  *         线性回归模型
  *
  **/
object LinearRegTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("线性回归模型").build
    Logger.getRootLogger.setLevel(Level.WARN)


    val file =LocalFile.file_root+"regression\\sample_linear_regression_data.txt"

    val data =MLUtils.loadLibSVMFile(sc,file)
    val randomSplit =  data.randomSplit(Array(0.8,0.2))
    val (train,test) = (randomSplit(0),randomSplit(1))

    val upStr="L1"
    val upter:Updater={
      upStr match {
        case "L1" => new L1Updater
        case "L2" => new SquaredL2Updater
        case _ => new SimpleUpdater
      }
    }

    val regParam:Double = 0.1
    val numIters:Int = 30
    val stepSize:Double =1.0


    val alg = new LinearRegressionWithSGD()
    alg.optimizer
        .setUpdater(upter)
      .setRegParam(regParam)
      .setNumIterations(numIters)
      .setStepSize(stepSize)

    val mod =alg.run(train)

    try{
      train.cache()

      val pre_lable = test.map{
        case (lp:LabeledPoint)=>{
          (mod.predict(lp.features),lp.label)
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
      val MSE:Double =pre_lable.map(t=>{
        math.pow( t._1- t._2 ,2)
      }).mean()

      val RMSE = math.sqrt(MSE)
      println(s"回归问题下:由线性回归(With SGD)模型预测的标准差为${RMSE}")


    }finally {
      train.unpersist(true)
    }


    sc.stop()
  }

}
