package com.hj.mllib.summary

import com.hj.util.{CorrelationMethodType, Params, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *相关性
  *相关系数的绝对值越大，相关性越强
  *
  * 0.8-1.0 极强相关
  * 0.6-0.8 强相关
  * 0.4-0.6 中等程度相关
  * 0.2-0.4 弱相关
  * 0.0-0.2 极弱相关或无相关
  **/
object Correlation {

  def main(args: Array[String]) {

    val file ="file:///E:\\idea-workspace\\ml\\sparkml\\deploy\\file\\mllib\\input\\regression\\sample_linear_regression_data.txt"

    val params =Params(input = file)


    val sc =SparkBuilder.appName("Correlation")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)


    val data = MLUtils.loadLibSVMFile(sc,params.input)
    data.cache()

    println(s"总数据量:${data.count()}")

    /*--------------------
    label 与各个 feature间的相关性
    --------------------------------------------------*/
    val labelRDD:RDD[Double] =data.map(_.label)
    val fs =data.map(_.features)

    /*-----------------------
    相关性系数：
    Supported: `pearson` (default), `spearman`
    ------------------------------------------*/

    val fs_size =  fs.first().size

    for(index <- 0 until  fs_size){

      /*逐个进行比对，获取相关性*/
      val featureRDD:RDD[Double] = fs.map(cols=> cols(index))

      val cor = Statistics.corr(labelRDD,featureRDD,CorrelationMethodType.pearson.toString)

      println(s"label列 与 该特征列的相关性系数为${cor}")

    }


    data.unpersist(true)
    sc.stop()
  }

}
