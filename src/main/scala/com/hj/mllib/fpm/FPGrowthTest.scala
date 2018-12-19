package com.hj.mllib.fpm

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  *         数据挖掘---寻找关联项
  *
  *         关联规则挖掘最典型的例子是购物篮分析，
  *
  *         通过分析可以知道哪些商品经常被一起购买，从而可以改进商品货架的布局。
  *
  *
  *         概念：
  *         1) 关联规则：用于表示数据内隐含的关联性，一般用X表示先决条件，Y表示关联结果。
  *
  *         (2) 支持度(Support)：所有项集中{X,Y}出现的可能性。
  *
  *         (3) 置信度(Confidence)：先决条件X发生的条件下，关联结果Y发生的概率。
  **/
object FPGrowthTest {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("FPGrowthTest")
      .build

    val file = LocalFile.file_root + "mllibFromSpark\\sample_fpgrowth.txt"

    val data: RDD[Array[String]] = sc.textFile(file).map(t => {
      val d = t.split(" ")
      d
    })


    /*--------------
    关联项寻找，购物篮分析--> 优化【货物架】的物品摆设
    -------------------------------------------------*/

    val minSupport: Double = 0.4 /*最小的支持度，小于该支持度的关联项，将会被移除*/

    val mod = new FPGrowth()
      .setMinSupport(minSupport)
      .setNumPartitions(2) //并发数
      .run(data)

    /*--------------输出符合支持度的数据-------------------------------*/

    val freqItemsets = mod.freqItemsets
    println(s"挖掘出的频繁项，子集有【${freqItemsets.count()}】个")

    mod.freqItemsets.foreach(itemsets => {
      println(itemsets.items.mkString("[", ",", "]") + "," + itemsets.freq)
    })

/*----------------------------------------------------------------
r z h k p
z y x w v u t s
s x o n r
x z y m t s q e
z
x z y r q t p
---------------------------------------------------------*/
/*-------------------------------------------------------------------
[t],3
[t,x],3
[t,x,z],3
[t,z],3
[y],3
[s],3
[s,x],3
[y,t],3
[z],5
[y,t,x],3
[y,t,x,z],3
[y,t,z],3
[y,x],3
[y,x,z],3
[y,z],3
[x],4
[x,z],3
[r],3
----------------------------------------------------*/
    sc.stop()
  }

}
