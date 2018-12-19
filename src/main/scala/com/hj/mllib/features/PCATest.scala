package com.hj.mllib.features

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         主成分分析（PCA） 是一种对数据进行旋转变换的统计学方法，
  *
  *         其本质是在线性空间中进行一个基变换，使得变换后的数据投影在一组新的“坐标轴”上的方差最大化，
  *
  *         随后，裁剪掉变换后方差很小的“坐标轴”，剩下的新“坐标轴”即被称为 主成分（Principal Component） ，
  *
  *         它们可以在一个较低维度的子空间中尽可能地表示原有数据的性质。
  *
  *         主成分分析被广泛应用在各种统计学、
  *
  *         机器学习问题中，是最常见的降维方法之一。
  *
  *         PCA有许多具体的实现方法，可以通过计算协方差矩阵，甚至是通过SVD分解来进行PCA变换。
  *
  **/
object PCATest {
  val file = LocalFile.file_root + "a.mat"

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("主成分析").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    /*----------------------

    MLlib提供了两种进行PCA变换的方法
    位于org.apache.spark.mllib.linalg包下的RowMatrix中

    其次：
    org.apache.spark.mllib.feature包下的PCA类
    -------------------------------------------*/
    //by_rowMatrix(sc)
    by_PCA_class(sc)

    sc.stop()
  }

  def by_rowMatrix(sc: SparkContext): Unit = {

    val data =genData(sc)

    val rowMatrix = new RowMatrix(data)

    /*----------------获取PCA------

    截取前3个
    ------------------------------*/

    val matrix = rowMatrix.computePrincipalComponents(3)

    println(matrix)

  }

  def genData(sc: SparkContext): RDD[Vector] = {

    val data: RDD[Vector] = sc.textFile(file).map(t => {
      val arr = t.split(" ").map(_.toDouble)
      Vectors.dense(arr)
    })

    data
  }

  def by_PCA_class(sc: SparkContext): Unit = {

    val data = genData(sc)

    val pca =new PCA(3)

    val mod = pca.fit(data)

    val rs = mod.transform(data)

    rs.foreach(v=> println(v.toString))

  }

}
