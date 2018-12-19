package com.hj.mllib.features

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         降维（Dimensionality Reduction） 是机器学习中的一种重要的特征处理手段，它可以
  *         消除噪声、对抗数据稀疏问题
  *
  *
  *         它在尽可能维持原始数据的内在结构的前提下，得到一组描述原数据的，低维度的隐式特征（或称主要特征）。
  *
  *         两个常用的降维方法：
  *         奇异值分解（Singular Value Decomposition，SVD）
  *         主成分分析（Principal Component Analysis，PCA）
  *
  **/
object SVDTest {

  def main(args: Array[String]) {
    val file = LocalFile.file_root + "a.mat"
    val sc = SparkBuilder.appName("主成分析").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    val data:RDD[LabeledPoint]=sc.textFile(file).map{ it=>{
      val arr = it.split(" ").map(_.toDouble)
      val labe :Double = if(arr(0)>1) 1.0 else 0.0
      LabeledPoint(labe,Vectors.dense(arr))
    }}

    /*---------------
    奇异值分解
    基于 RowMatrix 或者 IndexedRowMatrix实现
    --------------------------------------------*/

   val rowMatrix =new RowMatrix(data.map(_.features))


    val k =3 //number of leading singular values to keep
    val svd= rowMatrix.computeSVD(k,computeU = true)

    /*-----------------
    (4,9)矩阵--->  (9,3)

    s 向量
    v 右矩阵
    u,默认为空，只有当设置 computeU为 true，才可有值
    ----------------------------------------*/

    //println("s向量="+svd.s)
    //println("右矩阵:"+svd.V) //右矩阵，就是降维后的 目标数据

    println("（9,3）矩阵，转置成（3,9）")

    val rs1 =svd.V
    val rs2 = rs1.transpose

    //println(rs2)

    svd.U.rows.foreach(println)

    sc.stop()
  }

}
