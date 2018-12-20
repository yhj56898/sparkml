package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *         二分化
  **/
object BinarizerTest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("二值化").build
    val sql = new SQLContext(sc)

    /*------------------------------------模拟标签-----------------------------*/
    val data = Array((0, 0.1), (1, 0.8), (2, 0.2), (7, 0.5)).map(t => {
      t.copy(t._1.toDouble)
    })

    val df = sql.createDataFrame(data).toDF("label", "feature")

    val binarizerDF = new Binarizer()
      .setInputCol("label")
      .setOutputCol("binarizer_label")
      .setThreshold(2.0)
      .transform(df)

    /*--------------以给定是 阈值 为分割----------------------------------------------*/

    binarizerDF.show()

    sc.stop
  }

}
