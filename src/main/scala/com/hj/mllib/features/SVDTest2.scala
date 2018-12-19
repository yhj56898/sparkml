package com.hj.mllib.features

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  **/
object SVDTest2 {

  def main(args: Array[String]) {
    val file = LocalFile.file_root + "a.mat"
    val sc = SparkBuilder.appName("主成分析").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    val data: RDD[LabeledPoint] = sc.textFile(file).map { it => {
      val arr = it.split(" ").map(_.toDouble)
      val labe: Double = if (arr(0) > 1) 1.0 else 0.0
      LabeledPoint(labe, Vectors.dense(arr))
    }
    }
/*
* rows: RDD[IndexedRow]
* */
   val indexedRowMatrix= new IndexedRowMatrix(

      data.map(lp=>{
        IndexedRow(lp.label.toLong,lp.features)
      })

    )

    val svd =indexedRowMatrix.computeSVD(3,computeU = true)
    svd.U.rows.foreach(println)

    println(svd.V.transpose)

    sc.stop
  }

}
