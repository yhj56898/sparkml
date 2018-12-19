import com.hj.util.SparkBuilder

/**
  * @author hejin-Yu
  *
  *
  **/
object countByValueTest {

  def main(args: Array[String]) {

    val sc =SparkBuilder.appName("spark算子countByValue").build

    val  data=Map("k1"-> "v1",
      "k1"-> "v2",
      "k1"-> "v3",
      "k1"-> "v4","k1"-> "v5","k1"-> "v6",
      "k2"-> "v11"
    ).toArray

    val rdd =sc.parallelize(data)

    /*--------------countByValue---------------------------------------------------------*/

   val keyCounts = rdd.map{
      case (k,v)=>{
        k
      }
    }.countByValue()


    println(s"key的个数为${keyCounts.size}")


    /*--------------
    计算结果为 2

    说明，countByKey,已经做了去重操作
    ----------------------------------------------------*/

    sc.stop()
  }

}
