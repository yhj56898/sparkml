package com.hj.util

import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author hejin-Yu
  *
  *
  * 构建者模式，简化方法链的使用
  **/

class Builder(
               var master:String="local[*]",
               var appName:String="SparkBuilder"
             )

object SparkBuilder {
private val bulider = new Builder

  def master(master:String)= {
    bulider.master = master
    this
  }

  def appName(appName:String)={
    bulider.appName = appName
    this
  }

  def build:SparkContext={
    val conf =new SparkConf()
      .setMaster(bulider.master)
      .setAppName(bulider.appName)

    new SparkContext(conf)
  }


}
