
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.MalformedURLException
import java.net.URL
import java.util.stream.StreamSupport


fun main(args: Array<String>) {


}


fun doGet(httpurl: String?): String? {
    var connection: HttpURLConnection? = null
    var `is`: InputStream? = null
    var br: BufferedReader? = null
    var result: String? = null // 返回结果字符串
    try {
        // 创建远程url连接对象
        val url = URL(httpurl)
        // 通过远程url连接对象打开一个连接，强转成httpURLConnection类
        connection = url.openConnection() as HttpURLConnection?
        // 设置连接方式：get
        connection!!.requestMethod = "GET"
        // 设置连接主机服务器的超时时间：15000毫秒
        connection.connectTimeout = 15000
        // 设置读取远程返回的数据时间：60000毫秒
        connection.readTimeout = 60000
        // 发送请求
        connection.connect()
        // 通过connection连接，获取输入流
        if (connection.responseCode == 200) {
            `is` = connection.inputStream
            // 封装输入流is，并指定字符集
            br = BufferedReader(InputStreamReader(`is`, "UTF-8"))
            // 存放数据
            val sbf = StringBuffer()
            var temp: String? = null
            while (br.readLine().also { temp = it } != null) {
                sbf.append(temp)
                sbf.append("\r\n")
            }
            result = sbf.toString()
        }
    } catch (e: MalformedURLException) {
        e.printStackTrace()
    } catch (e: IOException) {
        e.printStackTrace()
    } finally {
        // 关闭资源
        if (null != br) {
            try {
                br.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        if (null != `is`) {
            try {
                `is`.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        connection!!.disconnect() // 关闭远程连接
    }
    return result
}