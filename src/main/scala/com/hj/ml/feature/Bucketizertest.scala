package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *         分桶
  *
  **/
object Bucketizertest extends LogUtil{

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("数据分桶").build
    val sql =new SQLContext(sc)

    val data =Array(-0.5,-0.3,0.0,0.2,0.7).map(x=> Tuple1.apply(x))

    val df = sql.createDataFrame(data).toDF("features")

    //val bucks:Array[Double] =Array(Double.NegativeInfinity,-0.5,0.5,Double.PositiveInfinity)

    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)

    val bucketizerDF =  new Bucketizer()
         .setInputCol("features")
         .setOutputCol("bucketizer_fs")
         .setSplits(splits) // [ ) -->前闭后开
        .transform(df)

    bucketizerDF.show

    sc.stop()
  }

}
