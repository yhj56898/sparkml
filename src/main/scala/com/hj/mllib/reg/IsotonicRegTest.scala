package com.hj.mllib.reg

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.regression.IsotonicRegression
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         保序回归
  *
  *         这种方法只有一个约束条件即，函数空间为单调递增函数的空间。基于reliability diagram，给定模型预测的分数，
  *
  *
  *         保序回归的应用之一就是用来做统计推断，
  *
  *         比如药量和毒性的关系，一般认为毒性随着药量是不减或者递增的关系，借此可以来估计最大药量
  *
  *
  *         广告投放上的应用
  *         万事万物，都遵循一个`度`,超过了这个度，都会失效、失序，陷入混沌，而保序回顾，探讨的就是这个`度`
  *
  **/
object IsotonicRegTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("保序回归").build

    val file =LocalFile.file_root+"mllibFromSpark\\sample_isotonic_regression_data.txt"


    val data:RDD[(Double,Double,Double)] = sc.textFile(file).map(x=>{
      val splits = x.split(",").map(_.toDouble)
      (splits(0),splits(1),1.0)
    })

    val randomSplits =data.randomSplit(Array(0.7,0.3))
    val (train,test)=(randomSplits(0),randomSplits(1))
    train.cache()
    /**
      * Run IsotonicRegression algorithm to obtain isotonic regression model.
      *
      * @param input RDD of tuples (label, feature, weight) where label is dependent variable
      *              for which we calculate isotonic regression, feature is independent variable
      *              and weight represents number of measures with default 1.
      *              If multiple labels share the same feature value then they are ordered before
      *              the algorithm is executed.
      * @return Isotonic regression model.
      */
    val mod =new IsotonicRegression().setIsotonic(true)
      .run(train)

    val pre_label = test.map{
      case (label,feature,w)=>{
        (mod.predict(feature),label)
      }
    }

    val mse = pre_label.map(t=>{
      math.pow( t._1- t._2 ,2)
    }).mean()

    println(s"mse:${mse}")

    train.unpersist(true)
    sc.stop()
  }

}
