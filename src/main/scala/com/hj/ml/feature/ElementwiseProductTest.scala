package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *
  *         给定一个 Vector，如权重
  *         选择一个 DataFrame列
  *         元素一一相乘
  **/
object ElementwiseProductTest extends LogUtil{

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("元素相乘").build

    val sql =new SQLContext(sc)

    val data =Array(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0))
    )

    val  df =sql.createDataFrame(data).toDF("id","features")

    val v =Vectors.dense(0.0, 1.0, 2.0)

    val rs =new ElementwiseProduct()
        .setInputCol("features")
        .setOutputCol("wiseFeatures")
        .setScalingVec(v) //给定一个向量
        .transform(df)

    rs.show()

    sc.stop
  }

}
