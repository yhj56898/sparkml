package com.hj.mllib.cluster

import com.hj.util.SparkBuilder
import org.apache.log4j.{Level, Logger}

/**
  * @author hejin-Yu
  *
  *
  *         幂迭代聚类
  *           基于图聚类
  **/
object PoweItClusterTest {

  def main(args: Array[String]) {
    val sc = SparkBuilder.appName("幂迭代聚类").build

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)



    sc.stop()
  }

}
