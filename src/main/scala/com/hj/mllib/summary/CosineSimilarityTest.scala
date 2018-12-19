package com.hj.mllib.summary

import com.hj.util.{LocalFile, Params, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * @author hejin-Yu
  *
  *         计算矩阵内 向量间的相关性,
  *
  **/
object CosineSimilarityTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("CosineSimilarityTest")
      .build

    Logger.getRootLogger.setLevel(Level.WARN)

    val file =LocalFile.file_root+"sample_svm_data.txt"
    val paras =Params(input = file)

    val data:RDD[Vector] =sc.textFile(paras.input).map(row=>{
      val d:Array[Double] =row.split(" ").map(_.toDouble)
      Vectors.dense(d)
    })
    data.cache()

    val mat =new RowMatrix(data)

    val exact =mat.columnSimilarities()

    val approx = mat.columnSimilarities(paras.threadHold)

    /*---------------------
    暴力求解
    使用阈值,求解
    ------------------------------------------------*/

    val exactEntries = exact.entries.map{
      case MatrixEntry(i,j,u)=>  ((i, j), u)
    }

    val approxEntries = approx.entries.map{
      case MatrixEntry(i,j,v)=>  ((i, j), v)
    }

    /*
    * 评估下，
    * 平均.绝对值.误差
    * */
    val MAE = exactEntries.leftOuterJoin(approxEntries)
        .values.map{
      case (u,Some(v)) => math.abs(u - v)
      case (u,None) => math.abs(u)
    }.mean()

    println(s"暴力求取余弦相似度、使用阈值0.1求取的相似度，两者间的绝对值-平均值为${MAE}")

    data.unpersist(true)
    sc.stop()
  }

}
