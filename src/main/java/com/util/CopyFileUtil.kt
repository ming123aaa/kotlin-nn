package com.util

import java.io.*
import java.lang.StringBuilder
import java.nio.charset.Charset

object CopyFileUtil {


    /**
     * 复制文本并替换
     */
    fun copyStringFileReplace(sourcePath: String, targetPath: String, call:(String)->String) {
        var fileReader = InputStreamReader(FileInputStream(File(sourcePath)),Charset.forName("UTF-8"))

        var sb = StringBuilder()


        var readLines = fileReader.readLines()
        readLines.forEach {
            sb.append(call(it)).append("\n")
        }


        var fileWriter = OutputStreamWriter(FileOutputStream(targetPath),Charset.forName("UTF-8"))
        // 把替换完成的字符串写入文件内
        fileWriter.write(sb.toString().toCharArray());
        // 关闭文件流，释放资源
        fileReader.close()
        fileWriter.close()
    }
}