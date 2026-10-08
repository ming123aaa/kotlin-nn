package com.model

import com.ai.Activations
import com.ai.NeuralNetwork
import com.ai.matrix
import com.ai.neuralNetwork
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/** 随机初始化权重矩阵，值域 (-1, 1) 并做 Xavier 缩放 */
private fun randomW(rows: Int, cols: Int) = matrix {
    val scale = Math.sqrt(2.0 / rows)
    for (r in 0 until rows) {
        row((0 until cols).map { Random.nextDouble(-scale, scale) })
    }
}

/** 初始化为全零的偏置矩阵 */
private fun zeroB(rows: Int, cols: Int) = matrix {
    for (r in 0 until rows) {
        row((0 until cols).map { 0.0 })
    }
}

/** 低体重	< 18.5
正常体重	18.5 – 24.9
超重	25.0 – 29.9
肥胖 I 级	30.0 – 34.9
肥胖 II 级	35.0 – 39.9
肥胖 III 级	≥ 40.0
 */
private fun bmiLabel(heightM: Double, weightKg: Double): Double {
    val bmi = weightKg / (heightM * heightM)
    return bmi / 100

}

/** 将标签转为 one-hot 向量 */
private fun labelToOneHot(label: Double): List<Double> {
    return listOf(label)
}


/** 生成数据集: 随机身高体重，用 BMI 公式计算标签并归一化 */
private fun generateDataset(count: Int): Pair<List<BimInputData>, List<Double>> {
    val xList = mutableListOf<BimInputData>()
    val yList = mutableListOf<Double>()
    for (i in 0 until count) {
        val h = Random.nextDouble(1.50, 1.90)   // 身高 1.50 ~ 1.90 m
        val w = Random.nextDouble(40.0, 110.0)   // 体重 40 ~ 110 kg
        val hNorm = (h - 1.50) / 0.40             // 归一化到 [0, 1]
        val wNorm = (w - 40.0) / 70.0             // 归一化到 [0, 1]
        val label = bmiLabel(h, w)
        xList.add(BimInputData(hNorm, wNorm))
        yList.add(label)
    }
    return Pair(xList, yList)
}



/** 计算分类准确率 */
private fun accuracy(preds: List<Double>, labels: List<Double>): Double {
    var correct = 0.0
    for (i in preds.indices) {
        val value = preds[i]
        val label = labels[i]
        val dt = 1.0 - min(max(abs(value - label) * 100, 0.1) - 0.1, 1.0)
        correct += sqrt(dt)

    }
    return correct.toDouble() / preds.size
}

fun main() {
    // ==================== 网络结构 ====================
    // 输入: [身高, 体重] (归一化后)
    // 输出: IBM*0.01


    val startTime = System.currentTimeMillis()
    val mBimModel= BimModel(neuralNetwork(2,1){
        neural(8,128, activation = Activations.RELU)



    })
    // ==================== 生成数据 ====================
    println("生成训练数据 (根据 BMI 公式)...")

    val (valX, valY) = generateDataset(3000)

    println("验证集: ${valX.size} 条")

    println()

    // ==================== 训练 ====================
    println("开始训练...")
    var pred = mBimModel.run(valX)
    var acc = accuracy(pred, valY)
    println("训练前  验证集准确率 = ${"%.2f".format(acc * 100)}%")
    var learningRate = 0.05


    for (epoch in 1..10) {
        val (trainX, trainY) = generateDataset(1000)
        if (learningRate > 0.001) {
            learningRate *= 0.9
        }


        mBimModel.train(trainX, trainY, learningRate, batchSize = 1)
        pred = mBimModel.forward(valX)
        acc = accuracy(pred, valY)
        println("逐样训练 $epoch 验证集准确率 = ${"%.2f".format(acc * 100)}%")


    }


    learningRate = 0.1

    for (epoch in 1..10) {
        val (trainX, trainY) = generateDataset(10000)

        if (learningRate > 0.0001) {
            learningRate *= 0.95
        }
        mBimModel.train(trainX, trainY, learningRate, batchSize = 500)


        val pred = mBimModel.forward(valX)
        val acc = accuracy(pred, valY)
        println("整批训练 $epoch  验证集准确率 = ${"%.2f".format(acc * 100)}%")

    }

    val tranTime = System.currentTimeMillis()
    println("训练时长:"+(tranTime-startTime)/1000.0)





    val accList = mutableListOf<Double>()
    for (epoch in 1..20) {
        val (valX, valY) = generateDataset(1000)
        val pred = mBimModel.forward(valX)
        val acc = accuracy(pred, valY)
        accList.add(acc)
        println("数据集${epoch} 验证集准确率 = ${"%.2f".format(acc * 100)}%")
    }
    val minAcc = accList.min()
    val avg = accList.average()
    val variance = accList.fold(0.0, { sum, acc -> (acc - avg) * (acc - avg) + sum }).div(accList.size)
    println("数据集平均准确率 = ${"%.2f".format(avg * 100)}%  最小=${"%.2f".format(minAcc * 100)}% 方差=${variance}")
    val testTime = System.currentTimeMillis()
    println("测试数据时长:"+(testTime - tranTime)/1000.0)

    pred = mBimModel.forward(valX)
    acc = accuracy(pred, valY)
    println("训练后  验证集准确率 = ${"%.2f".format(acc * 100)}%")

    // 导出模型
    val modelPath = "build/model/bmi_model.txt"
    mBimModel.save(modelPath)


    // 重新导入验证
    val loadedModel= BimModel.load(modelPath)
    val loadedPred = loadedModel.run(valX)
    val loadedAcc = accuracy(loadedPred, valY)
    println("导入后验证  验证集准确率 = ${"%.2f".format(loadedAcc * 100)}% (应与训练后一致)")

    // ==================== 评估 ====================
    println("\n==================== 测试示例 ====================")
    val testCases = listOf(
        Triple(1.65, 45.0, bmiLabel(1.65, 45.0)),   // 偏瘦
        Triple(1.70, 65.0, bmiLabel(1.70, 65.0)),   // 正常
        Triple(1.75, 90.0, bmiLabel(1.75, 90.0)),   // 超重
        Triple(1.60, 50.0, bmiLabel(1.60, 50.0)),   // 偏瘦
        Triple(1.80, 75.0, bmiLabel(1.80, 75.0)),   // 正常
        Triple(1.55, 85.0, bmiLabel(1.55, 85.0)),   // 超重
    )
    for ((h, w, expected) in testCases) {
        val hNorm = (h - 1.50) / 0.40
        val wNorm = (w - 40.0) / 70.0
        val input = BimInputData(hNorm, wNorm)
        val output = loadedModel.run(input)
        val predClass = output * 100
        val bmi = w / (h * h)
        val ok = if (abs(predClass - bmi) < 1) "✓" else "✗"
        println(
            "身高=%.2fm  体重=%.0fkg  BMI=%.1f  模型预测BIM=%s    $ok".format(
                h, w, bmi, predClass
            )

        )
    }
}

data class BimInputData(val height: Double, val weight: Double)
class BimModel(network: NeuralNetwork) : BaseModel<BimInputData, Double>(network) {
    override fun inputMatrix(input: BimInputData): DoubleArray {
         return doubleArrayOf(input.height, input.weight)
    }

    override fun outputMatrix(output: Double): DoubleArray {
        return doubleArrayOf(output)
    }

    override fun result(outMatrix: DoubleArray): Double {
        return outMatrix[0]
    }

    companion object{

        fun load(path: String): BimModel {
            return BimModel(NeuralNetwork.load(path))
        }

    }


}