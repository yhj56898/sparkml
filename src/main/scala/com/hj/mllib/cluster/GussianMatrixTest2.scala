package com.hj.mllib.cluster

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         上个例子，不能得到原来的label,
  *
  **/
object GussianMatrixTest2 {

  def main(args: Array[String]) {

    val k: Int = 3 //初始化有三个中心点
    /*鸢尾花，数据集中有三类：virginica、versicolor、setosa*/

    val file = LocalFile.file_root + "mllibFromSpark\\iris.data"

    val convergenceTol: Double = 0.01 //对数似然函数的收敛阈值,默认0.01


    val sc = SparkBuilder.appName("高斯混合聚类模型")
      .build


    val data: RDD[(String, Vector)] = sc.textFile(file).map(x => {
      val d = x.split(",") //.dropRight(1)// 5.1,3.5,1.4,0.2,Iris-setosa

      val f = d.dropRight(1).map(_.toDouble)
      val label = d(4)

      (label, Vectors.dense(f))
    })

    data.cache()

    val randomSplit = data.randomSplit(Array(0.8, 0.2))
    val (train, test) = (randomSplit(0), randomSplit(1))

    val numIters: Int = 20

    val cluster = new GaussianMixture()
      .setK(k)
      .setConvergenceTol(convergenceTol)
      .setMaxIterations(numIters)
      .run(train.map(_._2))


    val labelWithPre:RDD[(String,Int)] =test.map{
      case (label:String,vector:Vector)=>{
        (label,cluster.predict(vector))
      }
    }
    /*------------------------输出预测结果--------------------------------------*/

    labelWithPre.foreach(it=>{
      println(s"label:${it._1} \t preK:${it._2}")
    })

    data.unpersist(true)
    sc.stop()

  }

}
