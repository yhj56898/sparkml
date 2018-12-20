import java.util.UUID

/**
  * @author hejin-Yu
  *
  *
  **/
object UUIDTest {

  def main(args: Array[String]) {

    val uid = UUID.randomUUID().toString.takeRight(12)

    println(s"截断取数12位=${uid}")

  }

}
