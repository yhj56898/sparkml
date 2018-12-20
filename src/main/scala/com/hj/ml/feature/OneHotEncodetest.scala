package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *数据离散化 处理
  **/
object OneHotEncodetest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("独热编码").build
    val sql =new SQLContext(sc)

    val data =Array(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    val df =sql.createDataFrame(data).toDF("id","lable")



    val str2Index =new StringIndexer()
        .setInputCol("lable")
        .setOutputCol("str2Index")
        .fit(df)
      .transform(df)

    /*--------------------离散化 处理--------------------------*/

   val rs = new OneHotEncoder()
        .setInputCol("str2Index")
        .setOutputCol("oneHotencode_label")
        //.setDropLast()
        .transform(str2Index)

    rs.show(3,false)

    sc.stop()

  }

}
