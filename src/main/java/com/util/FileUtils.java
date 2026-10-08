package com.util;


import java.io.*;
import java.math.BigDecimal;


public class FileUtils {

    public static String readText(File file) throws FileNotFoundException {
        FileInputStream fis = null;
        try {
            fis = new FileInputStream(file);
            int data;

            System.out.println("可读取的字节数：" + fis.available());
            System.out.print("文件的内容为：");
            StringBuilder stringBuilder = new StringBuilder();
            while ((data = fis.read()) != -1) {
                stringBuilder.append((char) data);

                //read()方法返回的是整数，如果读取的是字符串，需要强制类型转换
            }
            return stringBuilder.toString();
        } catch (IOException ioe) {
            System.out.println("出现错误");
            ioe.getStackTrace();
        }
        try {
            fis.close();  //当流为空时不可关闭流，会发出空指针异常
        } catch (IOException ioe) {
            ioe.getStackTrace();
        }
        return null;

    }

    public static void writeText(File file, String s) {
        OutputStream fos = null;
        try {
            //构造一个文件输出流对象
            fos = new FileOutputStream(file);
            //此时的文件输出流对象fos就和目标源数据源（FileOutputStreamTest.txt文件）关联起来
            //利用文件输出流的方法把数据写入到文本中
            String str = s;
            byte[] words = str.getBytes();
            //利用write方法将数据写入到文件中去
            fos.write(words, 0, words.length);
        } catch (IOException ioe) {
            ioe.getStackTrace();
        }
        try {
            //关闭文件输出流
            fos.close();
        } catch (IOException ioe) {
            ioe.getStackTrace();
        }

    }

    public static long getSize(File file) {
        long size = 0;
        try {
            File[] fileList = file.listFiles();
            for (File f : fileList) {
                if (f.isDirectory()) {
                    size = size + getSize(f);
                } else {
                    size = size + f.length();
                }
            }
        } catch (Exception ignore) {
        }
        return size;
    }

    /**
     * 格式化单位
     */
    public static String formatSize(double size) {
        double kiloByte = size / 1024;
        if (kiloByte < 1) {
            return "0KB";
        }
        double megaByte = kiloByte / 1024;
        if (megaByte < 1) {
            BigDecimal result1 = new BigDecimal(Double.toString(kiloByte));
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).toPlainString() + "KB";
        }
        double gigaByte = megaByte / 1024;
        if (gigaByte < 1) {
            BigDecimal result2 = new BigDecimal(Double.toString(megaByte));
            return result2.setScale(2, BigDecimal.ROUND_HALF_UP).toPlainString() + "MB";
        }
        double teraBytes = gigaByte / 1024;
        if (teraBytes < 1) {
            BigDecimal result3 = new BigDecimal(Double.toString(gigaByte));
            return result3.setScale(2, BigDecimal.ROUND_HALF_UP).toPlainString() + "GB";
        }
        BigDecimal result4 = new BigDecimal(teraBytes);
        return result4.setScale(2, BigDecimal.ROUND_HALF_UP).toPlainString() + "TB";
    }

    public static boolean delete(File file) {
        if (file == null) {
            return false;
        }
        if (file.isDirectory()) {
            String[] children = file.list();
            for (String c : children) {
                boolean success = delete(new File(file, c));
                if (!success) {
                    return false;
                }
            }
        }
        return file.delete();
    }


}
