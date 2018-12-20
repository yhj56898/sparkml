package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *
  *         基于ML的主成分析
  *
  **/
object PCAtest extends LogUtil {

  def main(args: Array[String]) {
    val sc = SparkBuilder.appName("主成分析").build
    val sql = new SQLContext(sc)

    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val df =sql.createDataFrame(data.map(Tuple1.apply(_))).toDF("features")

    val pca =new PCA()
        .setK(5) //top N
        .setInputCol("features")
        .setOutputCol("pca_fs")
        .fit(df)


    pca.transform(df).show(false)

    sc.stop()
  }

}
