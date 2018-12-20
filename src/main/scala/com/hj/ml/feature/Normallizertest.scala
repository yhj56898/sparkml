package com.hj.ml.feature

import com.hj.util.{LocalFile, LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *
  **/
object Normallizertest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("数据规整化").build

    val sql =new SQLContext(sc)

    val file =LocalFile.file_root + "mllibFromSpark\\sample_libsvm_data.txt"

    val data =sql.read.format("libsvm").load(file)

    val nm =new Normalizer()
        .setInputCol("features")
        .setOutputCol("nm_features")
        .setP(1.0) //p 范数
        .transform(data)


    nm.show(3)




    sc.stop()
  }

}
