package com.hj.basic

import com.hj.util.SparkBuilder
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *索引化的行矩阵
  **/
object IndexedRowMatrixTest {

  def main(args: Array[String]) {

    val sc =SparkBuilder.appName("索引化的行矩阵").build

    val d1=Vectors.dense(1.0,2.0,3.0)

    val d2 =Vectors.dense(2.0,3.0,4.0)

    val indexedRow1 = IndexedRow(1,d1)
    val indexedRow2 = IndexedRow(2,d2)

    val rdd:RDD[IndexedRow] =sc.parallelize(Array(indexedRow1,indexedRow2))

   val rs = new IndexedRowMatrix(rdd)

    rs.rows.foreach(ir=>{
      println(ir)
    })

    sc.stop
  }

}
