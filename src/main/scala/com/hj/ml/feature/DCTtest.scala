package com.hj.ml.feature

import com.hj.util.SparkBuilder
import org.apache.spark.ml.feature.DCT
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *
  * 离散余弦 变换
  **/
object DCTtest {

  def main(args: Array[String]) {

    val sc =SparkBuilder.appName("离散余弦转换").build
    val sql=new SQLContext(sc)

    val data =Array(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0)
    )

    //DCT 转换
    val df =sql.createDataFrame(data.map(Tuple1.apply(_))).toDF("features")

    val dctDf =new DCT()
        .setInputCol("features")
        .setOutputCol("dct_features")
        .setInverse(false) //
        .transform(df)

    dctDf.show(false)


    sc.stop
  }

}
