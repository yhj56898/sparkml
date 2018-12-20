package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *         将标签，诸如 男、女 ，有、无 。。。
  *
  *         ---------------> 转成数字
  *         。。。。当然，也可以提前处理好
  *         如 case sex when '男' then 0.0 else 1.0
  *
  *         indexToString ---> 需要与StringIndexer成对出现
  **/
object Indexer2stringTest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("indexer2String").build

    val sql = new SQLContext(sc)

    val data = Array(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    val df = sql.createDataFrame(data).toDF("id", "label")


    val string2indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("str2Indexer_label")
      .fit(df)
      .transform(df)


    /*--------------------成对出现 IndexerToString------------------------------*/
    val indexer2Str = new IndexToString()
      .setInputCol("str2Indexer_label")
      .setOutputCol("ori_label")
      .transform(string2indexer)

    indexer2Str.show(false)

    sc.stop()
  }

}
