package com.hj.mllib.reg

import com.hj.util.{LocalFile, Params, SparkBuilder}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         决策树回归模型
  **/
object DecisionTreeRegTest {

  def main(args: Array[String]) {

    val file = LocalFile.file_root + "classification\\sample_libsvm_data.txt"
    val paras = Params(input = file)

    val sc = SparkBuilder.appName("DecisionTreeRegTest")
      .build

    val data = MLUtils.loadLibSVMFile(sc, paras.input)

    data.cache()

    val randomSplit: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.8, 0.2))

    val (train, test) = (randomSplit(0), randomSplit(1))

    /*----------------------基于决策树，做回归预测模型------------*/

    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity: String = "variance"
    /*Supported values: "variance"*/
    val maxDepth: Int = 5
    /* (suggested value: 5)*/
    val maxBins: Int = 32 /*(suggested value: 32)*/


    val mod = DecisionTree.trainRegressor(train,
      categoricalFeaturesInfo, impurity, maxDepth, maxBins)


    val preWithLabel = test.map {
      case (LabeledPoint(label, features)) => {
        (mod.predict(features), label)
      }
    }

    /*平均.平方.误差*/

    val MSE = preWithLabel.map {
      case (pre, lab) => math.pow(pre - lab, 2)
    }.mean()


    println(s"Mean Squared Error:${MSE}")


    data.unpersist(true)
    sc.stop()
  }

}
