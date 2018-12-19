

/**
  * @author hejin-Yu
  *
  *
  **/
object TestArrayDrop {

  def main(args: Array[String]) {

    val str="5.1,3.5,1.4,0.2,Iris-setosa"

    val data =str.split(",").dropRight(1)


    println(data.mkString(","))

  }

}
