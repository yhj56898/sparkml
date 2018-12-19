package com.hj.mllib.cluster

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         混合高斯模型
  *
  *         混合高斯模型的应用场景包括：
  *         数据集分类，如会员分类；
  *
  *         图像分割以及以及特征抽取，
  *             例如在视频中跟踪人物以及区分动作，识别汽车、建筑物等；
  *
  *         语音分割以及特征特征抽取，
  *           例如从一堆杂乱的声音中提取某个人的声音，从音乐中提取背景音乐，从大自然中提取地震的声音等。
  **/
object GaussianMixtureTest {

  def main(args: Array[String]) {

    /*-----------
    高斯混合模型（Gaussian Mixture Model, GMM） 是一种概率式的聚类方法，属于生成式模型，
    它假设所有的数据样本都是由某一个给定参数的 多元高斯分布 所生成的

    具体地，依赖给定的 类个数K
    ----------------------------------------------------------------*/

    val k:Int = 3 //初始化有三个中心点
    /*鸢尾花，数据集中有三类：virginica、versicolor、setosa*/

    val file=LocalFile.file_root+"mllibFromSpark\\iris.data"

    val convergenceTol:Double=0.01 //对数似然函数的收敛阈值,默认0.01


    val sc =SparkBuilder.appName("高斯混合聚类模型")
      .build


    val data:RDD[Vector] =sc.textFile(file).map(x=>{
      val  d  =x.split(",").dropRight(1)// 5.1,3.5,1.4,0.2,Iris-setosa
          .map(_.toDouble)
      Vectors.dense(d)
    })

    data.cache()

    val randomSplit = data.randomSplit(Array(0.8,0.2))
    val (train,test ) =(randomSplit(0),randomSplit(1))

    val numIters:Int = 20

    val cluster = new GaussianMixture()
      .setK(k)
      .setConvergenceTol(convergenceTol)
      .setMaxIterations(numIters)
      .run(train)

   val rs = cluster.predict(test)

    /*------------------------输出预测结果--------------------------------------*/

    rs.foreach(t=>{
      println(s"预测值=${t}")
    })

    data.unpersist(true)
    sc.stop()

  }

}
