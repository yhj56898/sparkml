package com.hj.mllib.fpm

import com.hj.util.SparkBuilder
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.rdd.RDD
/**
  * @author hejin-Yu
  *
  *
  * 关联规则
  *
  *   反映一个事物与其他事物之间的相互依存性和关联性
  *
  *     分析出形如“由于某些事件的发生而引起另外一些事件的发生”之类的规则
  *
  *
  *
  *     价目表设计、商品促销、商品的排放和基于购买模式的顾客划分
  **/
object AssociationRulesTest {

  def main(args: Array[String]) {

    val sc=SparkBuilder.appName("关联规则").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    /*--------------------
    基于协同的 过滤推荐 ---> 也可以挖掘事物间的 关联项
      但是，它依赖个前期的数据准备 与 特定的 1~5 5个频度的分数项
      也就是评判系统

      关联规则，应该是暴力求解的过程，省去了前期的 评分过程
    -------------------------------------------------*/

/*    val freqItemsets = sc.parallelize(Seq(
       new FreqItemset(Array("a"),15L), //a,出现的次数 15
      new FreqItemset[String](Array("b"),32L), //比元素，出现的次数为 32
      new FreqItemset[String](Array("c"),32L), //比元素，出现的次数为 32
      new FreqItemset[String](Array("d"),32L), //比元素，出现的次数为 32
      new FreqItemset[String](Array("a","b"),12),//元素a,b一起出现的次数 12
      new FreqItemset[String](Array("a","c"),20),//元素a,b一起出现的次数 12
      new FreqItemset[String](Array("a","c","d"),20)//元素a,b一起出现的次数 12
    ))
    */

    val freqItemsets =sc.parallelize(Seq(
      new FreqItemset(Array("z"),6),
/*      new FreqItemset(Array("z","y","x","w","v","u","t","s"),1),
      new FreqItemset(Array("w"),1),
      new FreqItemset(Array("r","z","h","k","p"),1),
      new FreqItemset(Array("s"),3),
      new FreqItemset(Array("p"),2),
      new FreqItemset(Array("e"),1),
      new FreqItemset(Array("x"),4),
      new FreqItemset(Array("k"),1),
      new FreqItemset(Array("t"),3),
      new FreqItemset(Array("h"),1),
      new FreqItemset(Array("n"),1),
      new FreqItemset(Array("v"),1),
      new FreqItemset(Array("r"),3),*/
      new FreqItemset(Array("y"),3),
      new FreqItemset(Array("z","y"),3)
    ))


    /*-----------------
    挖掘两者间的关联性

    依赖一个给定的 置信度 【这个，就比较类似于协同过滤的算法了】
    --------------------------------------------*/

    val ar = new AssociationRules() //默认置信度为0.8
      .setMinConfidence(0.1) //,或者显示给定

   val rs:RDD[Rule[String]]  = ar.run(freqItemsets)

    rs.foreach(rules=>{
      println(rules)
    })

    sc.stop
  }

}
