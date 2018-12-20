package com.hj.ml.feature

import com.hj.util.{LogUtil, SparkBuilder}
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SQLContext

/**
  * @author hejin-Yu
  *         文本特征提取
  *
  *         CountVectorizer
  *         只考虑词汇在文本中出现的频率
  *
  *
  *         TfidfVectorizer
  *         除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
  *         能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征
  **/
object CountVectorizerTest extends LogUtil{

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("词频统计-特征选择").build

    val sql =new SQLContext(sc)

    val data = Array(
      (0,Array("a", "b", "c")),
      (1,Array("a", "b", "b", "c", "a"))
    )

    val df =sql.createDataFrame(data).toDF("id","words")

    val mode =new CountVectorizer()
        .setInputCol("words")
        .setOutputCol("wordsCntFeatures")
        .setMinDF(2) //文档中最小的 出现频次
        .setVocabSize(2) //取top N,数据的张量
        .fit(df)


    mode.transform(df).show

    sc.stop()

  }

}
