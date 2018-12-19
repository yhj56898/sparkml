package com.hj.basic

import org.apache.spark.mllib.linalg.Vectors


/**
  * @author hejin-Yu
  *
  * 向量
  *
  * 稠密向量DenseVector  稀疏向量SparseVector
  *
  * 一堆的double---> DenseVector
  * (3,[0,2],[1.0,3.0]) --> 缺省角标，值即为0
  *
  **/
object VectorTest {

  def main(args: Array[String]) {

    val denseVector = Vectors.dense(Array(2.0,0.0,8.0))

    val sparseVector =Vectors.sparse(3,Array(0,2),Array(1.0,3.0))


    println(denseVector)
    println(sparseVector)


  }

}
