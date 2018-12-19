import com.hj.util.SparkBuilder

/**
  * @author hejin-Yu
  *
  *
  **/
object MeanTest {

  def main(args: Array[String]) {

    val data= 1 to 10

    val totalRs:Int =data.sum
    val nums = data.length

    val means1:Double = totalRs.toDouble / nums


    val sc =SparkBuilder.appName("Means")
      .build

    val means2 = {

      val rdd =sc.parallelize(data)
      rdd.mean()
    }

    println(s"means1:${means1}")
    println(s"means2:${means2}")


    sc.stop()
  }

}
