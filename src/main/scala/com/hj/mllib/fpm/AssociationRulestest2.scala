package com.hj.mllib.fpm

import com.hj.util.{LocalFile, SparkBuilder}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.rdd.RDD

/**
  * @author hejin-Yu
  *
  **/
object AssociationRulestest2 {

  def main(args: Array[String]) {
    val sc = SparkBuilder.appName("关联规则").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val file = LocalFile.file_root + "mllibFromSpark\\sample_fpgrowth.txt"

    val data = sc.textFile(file)

    /*-----------拆成 （k,出现的次数）------------------------------------------*/

    val k_cnts = data.flatMap(_.split(" "))
      .map(x => (x, 1))
      .reduceByKey(_ + _)

    val old = data.map(t => {
      (t, 1)
    }).reduceByKey(_ + _)

    val rs_items = (k_cnts ++ old)
      .reduceByKey(_ + _)


    /*------------------构建FreqItemset----------------------------------*/

    val freqItemset: RDD[FreqItemset[String]] = rs_items.map(it => {
      new FreqItemset[String](Array(it._1), it._2.toLong)
    })



    freqItemset.foreach(t=>{
      println(t.items.mkString(" ") +"\t"+t.freq)
    })

    val ar = new AssociationRules()//.setMinConfidence(0.01) //设置为0.8时，无输出
    /*--------------------------数据貌似有问题，不能输出-------------------------------------------------*/
    val rs:RDD[Rule[String]]  = ar.run(freqItemset)

    rs.foreach(rules=>{
      println(rules)
    })

    sc.stop()
  }

}
