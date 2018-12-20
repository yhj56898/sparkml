package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *         卡方选择器 ---特征选择
  *
  *
  *         特征选择（Feature Selection）指的是在特征向量中选择出那些“优秀”的特征，组成新的、更“精简”的特征向量的过程。
  *
  *         它在高维数据分析中十分常用，可以剔除掉“冗余”和“无关”的特征，提升学习器的性能。
  *
  *
  *         它通过对特征和真实标签之间进行卡方检验，来判断该特征和真实标签的关联程度，进而确定是否对其进行
  *         选择
  **/
object ChisqSelectorTest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("卡方选择").build

    val sql = new SQLContext(sc)

    val data = Array(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )

    val df = sql.createDataFrame(data).toDF("id", "features", "clicked")

    val rs = new ChiSqSelector()
      .setLabelCol("clicked")
      .setFeaturesCol("features")
      .setNumTopFeatures(1)
      .setOutputCol("chiseSelectorFeature")
      .fit(df)

    rs.transform(df).show

    sc.stop

  }

}
