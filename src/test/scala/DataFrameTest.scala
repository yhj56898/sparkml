import com.hj.util.{LocalFile, LogUtil, SparkBuilder}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.mllib.linalg.Vector

/**
  * @author hejin-Yu
  *
  *
  **/
object DataFrameTest extends LogUtil {

  def main(args: Array[String]) {

    val sc = SparkBuilder.appName("DataFrame测试").build
    val sql = new SQLContext(sc)

    val file = LocalFile.file_root + "mllibFromSpark\\sample_libsvm_data.txt"

    val libsvm = sql.read.format("libsvm").load(file).cache()
    /*------------------
     |-- label: double (nullable = false)
     |-- features: vector (nullable = false)
    -----------------------------------------------------------*/

//    val labelSummary =libsvm.describe("label") //基于label列，获取统计信息
//    labelSummary.show

    val fs= libsvm.select("features").map{
      case Row(v:Vector)=>{
        v
      }
    }


    libsvm.unpersist(true)
    sc.stop()

  }

}
