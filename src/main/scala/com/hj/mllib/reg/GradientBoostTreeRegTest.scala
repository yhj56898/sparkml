package com.hj.mllib.reg

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy}
import org.apache.spark.mllib.tree.configuration.Algo.Algo
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  * @desc
  *
  *      梯度上升树(分类的话，仅支持二分类)
  *        还能用来做回归
  **/
object GradientBoostTreeRegTest {

  def main(args: Array[String]): Unit = {

    val sc =SparkBuilder.appName("梯度上升树-回归问题探讨").build

    val file= LocalFile.file_root+"regression\\sample_libsvm_data.txt"

    Logger.getRootLogger.setLevel(Level.WARN)

    val data:RDD[LabeledPoint] =MLUtils.loadLibSVMFile(sc,file)
    data.cache()

    val randomSplit = data.randomSplit(Array(0.7,0.3))
    val (train,test) = (randomSplit(0),randomSplit(1))

    val method:Algo = Algo.Regression
    val numIters:Int = 20

    val boostStrategy:BoostingStrategy = BoostingStrategy.defaultParams(method)
    boostStrategy.numIterations =numIters
    boostStrategy.treeStrategy.maxDepth =5

    val mod =GradientBoostedTrees.train(train,boostStrategy)

    val pre_label= test.map{
      case (lp: LabeledPoint)=>{
        (lp.label,mod.predict(lp.features))
      }
    }

    /*----------------输出下MSE RMSE----------------------------------*/
    val mse =  pre_label.map(it=>{
      math.pow( it._1 - it._2 ,2)
    }).mean()

    val RMSE = math.sqrt(mse)

    println(s"MSE=${mse}\tRMSE=${RMSE}")

    data.unpersist(true)
    sc.stop()
  }
}
