package com.hj.util

import com.hj.util.Algorithm.Algorithm
import com.hj.util.RegType.RegType

/**
  * @author hejin-Yu
  *
  *
  **/
case class Params(
                   input: String = null, //输入
                   num_iters: Int = 20,
                   step_size: Double = 1.0,
                   algorithm: Algorithm = Algorithm.SVM,
                   regType: RegType = RegType.L1,
                   regPara: Double = 0.000001,
                   threadHold:Double=0.1
                 )



object Algorithm extends Enumeration {
  type Algorithm = Value
  val SVM, LR = Value
}

object RegType extends Enumeration {
  type RegType = Value
  val L1, L2, None = Value
}
object CorrelationMethodType extends Enumeration {
  type CorrelationMethodType =Value
  val pearson,spearman = Value
}
object LocalFile{
  val file_root="file:///E:\\idea-workspace\\ml\\sparkml\\deploy\\file\\mllib\\input\\"
}