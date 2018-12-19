package com.hj.mllib.claz

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.Algo.Algo
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         梯度上升树
  *         ---分类
  *         梯度上升树，仅支持二分类
  *
  *          We support regression and binary
  *                          classification for boosting
  *
  **/
object GradientBoostTreeClazTest {

  def main(args: Array[String]) {

/*-------------------鸢尾花，既定的数据类 有3类----------------------------------------*/
    val file=LocalFile.file_root+"mllibFromSpark\\iris.data"

    val sc =SparkBuilder.appName("GradientBoostTreestest")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)

    val data:RDD[LabeledPoint] =sc.textFile(file) //MLUtils.loadLibSVMFile(sc,file)
        .mapPartitions(its=>{
      its.map(it=>{
        val arr =it.split(",")


        val fs =arr.dropRight(1).map(_.toDouble)
        LabeledPoint(encode(arr(4)),Vectors.dense(fs))
      })
    })


    data.cache()

    val randomSplit =data.randomSplit(Array(0.8,0.2))
    val (train,test) = (randomSplit(0),randomSplit(1))

    val numClaz = 2
    val numIters:Int = 20
    val maxDepth = 5

    val boostingStrategy =BoostingStrategy.defaultParams(Algo.Classification)
    boostingStrategy.treeStrategy.numClasses =numClaz
    boostingStrategy.numIterations = numIters
    /*树深
    *
    * maxDepth：限定决策树的最大可能深度。但由于其它终止条件或者是被剪枝的缘故，最终的决策树的深度可能要比maxDepth小
    * */
    boostingStrategy.treeStrategy.maxDepth =maxDepth

    val mod = GradientBoostedTrees.train(train,boostingStrategy)


    val pre_lable:RDD[(Double,Double)] =test.map{
      case (labeledPoint: LabeledPoint)=>{
        (mod.predict(labeledPoint.features), labeledPoint.label)
      }
    }

    val rs= pre_lable.mapPartitions(its=>{
      its.map(it =>{
        (decode(it._1),decode(it._2))
      })
    })

    println("---------打印下")
    rs.foreach(it=> println(s"预测值=${it._1}\t真实值=${it._2}"))

    /*平方 平均*/
    val mse =  pre_lable.map(it=>{
     math.pow(it._1-it._2,2)
    }).mean()

    val RMSE = math.sqrt(mse) /*均方根误差*/

    println(s"模型预测的MSE=${mse}")
    println(s"模型预测的RMSE=${RMSE}")

    sc.stop()
  }

  /*-----------------编码---------------------------------------------------*/
  def encode(str:String):Double={
    str match {
      case "Iris-setosa" => 0.0
      //case "Iris-versicolor" => 1.0
      case _ => 1.0
    }
  }

  /*-----------------反编码---------------------------------------------------*/
  def decode(dou:Double):String={
    dou match {
      case 0.0 => "Iris-setosa"
      //case 1.0 => "Iris-versicolor"
      case _ => "Iris-virginica"
    }
  }
}
