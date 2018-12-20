package com.hj.util

import org.apache.log4j.{Level, Logger}
/**
  * @author hejin-Yu
  *
  *
  **/
trait LogUtil {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
}
