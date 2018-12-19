package com.hj.mllib.claz

import com.hj.util.{LocalFile, Params, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *
  **/
object DecisionTreeClassificationTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("DecisionTreeClassificationTest")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)

    val file = LocalFile.file_root + "classification\\sample_libsvm_data.txt"

    val paras = Params(input = file)

    val data = MLUtils.loadLibSVMFile(sc, paras.input)

    data.cache()

    val randomSplit: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.8, 0.2))

    val (train, test) = (randomSplit(0), randomSplit(1))


    val numsClzs: Int = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity: String = "gini"

    val max_Depth: Int = 5 //Maximum depth of the tree.  (suggested value: 5)
    val max_bins: Int = 32 //maximum number of bins used for splitting features

    /*------------- --- impurity
    case "gini" => Gini
    case "entropy" => Entropy
    case "variance" => Variance

    在决策树分类模型中，不支持variance增益方式
    -------------------------------------------------*/

    val mod = DecisionTree.trainClassifier(train, numsClzs,
      categoricalFeaturesInfo, impurity, max_Depth, max_bins)

    val preWithLabel = test.map{
      case (LabeledPoint(label, features))=>{
        (mod.predict(features),label)
      }
    }

    /*-----------------------准确率------------------------------------------*/
    val actual =preWithLabel.filter{
      case (pre,lab)=> pre == lab //准确
    }.count.toDouble / test.count()

    println(s"预测是正确率为${actual}")


    data.unpersist(true)
    sc.stop
  }

}
