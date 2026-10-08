package com.ai

import kotlin.random.Random
import java.util.Random as JRandom

// 激活函数: forward 正向计算, backward 计算导数 (传入 z，返回 σ'(z))
class Activation(
    val name: String,
    val forward: (Matrix) -> Matrix,
    val backward: (Matrix) -> Matrix
)

/** 内置激活函数 */
object Activations {
    /** Sigmoid: σ(z) = 1 / (1 + e^(-z))，导数 σ'(z) = σ(z) * (1 - σ(z)) */
    val SIGMOID = Activation(
        "SIGMOID",
        forward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    1.0 / (1.0 + Math.exp(-z.getValue(i, j)))
                }
            })
        },
        backward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    val s = 1.0 / (1.0 + Math.exp(-z.getValue(i, j)))
                    s * (1.0 - s)
                }
            })
        }
    )

    /** ReLU: f(z) = max(0, z)，导数 f'(z) = 1(if z > 0) else 0 */
    val RELU = Activation(
        "RELU",
        forward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    val v = z.getValue(i, j)
                    if (v > 0) v else 0.0
                }
            })
        },
        backward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    if (z.getValue(i, j) > 0) 1.0 else 0.0
                }
            })
        }
    )

    val LINEAR = Activation(
        "LINEAR",
        forward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    val v = z.getValue(i, j)
                    v
                }
            })
        },
        backward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { j ->
                    1.0
                }
            })
        })

    /** Softmax: 将输出转为概率分布，通常用于多分类输出层
     *  注意: 配合交叉熵损失时联合梯度 dL/dz = pred - label，
     *  因此 backward 返回全 1 矩阵，直接透传已计算好的 dL/dz */
    val SOFTMAX = Activation(
        "SOFTMAX",
        forward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                val row = (0 until z.columnCount()).map { j -> Math.exp(z.getValue(i, j)) }
                val sum = row.sum()
                row.map { it / sum }
            })
        },
        backward = { z ->
            matrix((0 until z.rowCount()).map { i ->
                (0 until z.columnCount()).map { 1.0 }
            })
        }
    )
}

private val activationMap = mapOf(
    "SIGMOID" to Activations.SIGMOID,
    "RELU" to Activations.RELU,
    "SOFTMAX" to Activations.SOFTMAX,
    "LINEAR" to Activations.LINEAR
)

// 单个神经层: 正向 linear + activation，反向传播梯度，梯度下降更新
class NerveNode(
    var wList: Matrix,            // 权重矩阵 W [输入维度 × 输出维度]
    var bList: Matrix,            // 偏置 b [1 × 输出维度]
    val activation: Activation = Activations.SIGMOID // 激活函数，默认 Sigmoid
) {
    private var inputCache: Matrix? = null  // 缓存上层的输出 x，反向传播时用于计算 dW
    private var zCache: Matrix? = null      // 缓存线性结果 z = x * W + b，反向传播时用于计算激活函数导数
    private var dW: Matrix? = null           // 权重的梯度
    private var dB: Matrix? = null           // 偏置的梯度

    /** 正向传播: output = activation(input * W + b) */
    fun forward(data: Matrix): Matrix {
        inputCache = data
        val linear = data * wList
        // 如果 bList 只有 1 行但 linear 有多行 (批处理)，则广播到相同行数
        val z = if (bList.rowCount() == 1 && linear.rowCount() > 1) {
            val tiled = (0 until linear.rowCount()).map { bList.getRow(0) }
            linear + matrix(tiled)
        } else {
            linear + bList
        }
        zCache = z
        return activation.forward(z)
    }

    /** 反向传播: 接收来自上一层的梯度 dL/da，返回传递给下一层的梯度
     *  dZ         = dLoss ⊙ activation'(z)        -- 哈达玛积
     *  dW         = input^T × dZ                  -- 权重梯度
     *  dB         = sum(dZ, axis=0)               -- 偏置梯度 (沿样本求和)
     *  dInput     = dZ × W^T                      -- 传给上一层的梯度
     */
    fun backward(dLoss: Matrix): Matrix {
        val dZ = dLoss.elementWiseTimes(activation.backward(zCache!!))
        dW = inputCache!!.transpose() * dZ
        dB = dZ.sumRows()
        return dZ * wList.transpose()
    }

    /** 梯度下降更新参数: W = W - lr * dW, b = b - lr * dB */
    fun update(learningRate: Double) {
        wList = wList.minus(dW!! * learningRate)
        bList = bList.minus(dB!! * learningRate)
    }
}

/** 权重初始化器
 *  @param weight  (fanIn, fanOut) → 权重矩阵
 *  @param bias    (1, fanOut) → 偏置矩阵
 */
class Initializer(
    val weight: (Int, Int) -> Matrix,
    val bias: (Int, Int) -> Matrix
)

object Initializers {
    private val gaussian = JRandom()

    /** He 初始化: N(0, sqrt(2/fanIn)), 适用于 ReLU */
    val HE = Initializer(
        weight = { inDim, outDim ->
            val scale = Math.sqrt(2.0 / inDim)
            matrix((0 until inDim).map { (0 until outDim).map { gaussian.nextGaussian() * scale } })
        },
        bias = { _, outDim -> matrix(listOf((0 until outDim).map { 0.0 })) }
    )

    /** Xavier 初始化: N(0, sqrt(2/(fanIn+fanOut))), 适用于 Sigmoid/Tanh */
    val XAVIER = Initializer(
        weight = { inDim, outDim ->
            val scale = Math.sqrt(2.0 / (inDim + outDim))
            matrix((0 until inDim).map { (0 until outDim).map { gaussian.nextGaussian() * scale } })
        },
        bias = { _, outDim -> matrix(listOf((0 until outDim).map { 0.0 })) }
    )
}

/** 快速创建神经网络
 *  @param inputDim    输入维度
 *  @param outputDim   输出维度
 *  @param layer       隐藏层层数
 *  @param size        每层隐藏层神经元数
 *  @param initializer 权重初始化器，默认 He 初始化
 *
 *  示例: neuralNetwork(2, 3, 2, 16)
 *  → 2输入 → 16(ReLU) → 16(ReLU) → 3(Softmax/分类)
 */
fun neuralNetwork(
    inputDim: Int,
    outputDim: Int,
    layer: Int,
    size: Int,
    initializer: Initializer = Initializers.HE,
    activation: Activation = Activations.RELU
): NeuralNetwork {
    val layers = mutableListOf<NerveNode>()
    var inDim = inputDim
    // 隐藏层
    for (i in 0 until layer) {
        layers.add(
            NerveNode(
                initializer.weight(inDim, size),
                initializer.bias(inDim, size),
                activation
            )
        )
        inDim = size
    }
    // 输出层
    val outActivation = if (outputDim > 1) Activations.SOFTMAX else Activations.SIGMOID
    layers.add(
        NerveNode(
            initializer.weight(inDim, outputDim),
            initializer.bias(inDim, outputDim),
            outActivation
        )
    )
    return NeuralNetwork(layers)
}

fun neuralNetwork(
    inputDim: Int,
    outputDim: Int,
    initializer: Initializer = Initializers.HE,
    block: NeuralNetworkBuilder.() -> Unit
): NeuralNetwork {
    return NeuralNetworkBuilder(inputDim, outputDim, initializer).also(block).build()
}

class NeuralNetworkBuilder(
    inputDim: Int,
    private val outputDim: Int,
    private val defaultInitializer: Initializer = Initializers.HE
) {
    private val layers = mutableListOf<NerveNode>()
    private var inDim = inputDim

    fun neural(
        layer: Int,
        size: Int,
        initializer: Initializer = Initializers.HE,
        activation: Activation = Activations.RELU
    ) {
        for (i in 0 until layer) {
            layers.add(
                NerveNode(
                    initializer.weight(inDim, size),
                    initializer.bias(inDim, size),
                    activation
                )
            )
            inDim = size
        }
    }

    fun build(): NeuralNetwork {
        val outActivation = if (outputDim > 1) Activations.SOFTMAX else Activations.SIGMOID
        val outList = layers.toMutableList()

        outList.add(
            NerveNode(
                defaultInitializer.weight(inDim, outputDim),
                defaultInitializer.bias(inDim, outputDim),
                outActivation
            )
        )
        return NeuralNetwork(outList)
    }
}


// 神经网络: 将多个 NerveNode 串联，提供 forward 和 train 接口
class NeuralNetwork(private val layers: List<NerveNode>) {

    /** 正向传播 (矩阵输入) */
    fun forward(input: Matrix): Matrix {
        return layers.fold(input) { data, layer -> layer.forward(data) }
    }

    /** 正向传播 (一维数组输入，返回一维数组) */
    fun forward(input: DoubleArray): DoubleArray {
        return forward(Matrix(arrayOf(input))).getRow(0)
    }

    /** 训练一步 (矩阵输入, 批量) */
    fun trainMatrix(input: Matrix, label: Matrix, learningRate: Double = 0.1) {
        val predicted = forward(input)
        // MSE 平均梯度: dL/dpred = (pred - label) / N，保证 batch size 变化时有效学习率一致
        val loss = predicted.minus(label) / input.rowCount().toDouble()
        var gradient = loss
        for (i in layers.size - 1 downTo 0) {
            gradient = layers[i].backward(gradient)
        }
        for (layer in layers) {
            layer.update(learningRate)
        }
    }


    /**
     * 逐样训练
     */
    fun trains(input: Matrix, label: Matrix, learningRate: Double = 0.1) {
        trainMiniBatch(input, label, learningRate, 1)
    }

    fun trainMiniBatch(input: Matrix, label: Matrix, learningRate: Double = 0.1, batchSize: Int = 64) {
        val totalRows = input.rowCount()
        var start = 0
        while (start < totalRows) {
            val end = (start + batchSize).coerceAtMost(totalRows)
            val batchInput = matrix((start until end).map { input.getRow(it) })
            val batchLabel = matrix((start until end).map { label.getRow(it) })
            trainMatrix(batchInput, batchLabel, learningRate)
            start = end
        }
    }

    /** 训练一步 (一维数组输入，单样本) */
    fun train(input: List<Double>, label: List<Double>, learningRate: Double = 0.1) {
        trainMatrix(matrix(listOf(input.toList())), matrix(listOf(label.toList())), learningRate)
    }

    /** 导出模型到文件 */
    fun save(filePath: String) {
        java.io.File(filePath).parentFile.mkdirs()
        val sb = StringBuilder()
        sb.appendLine(layers.size.toString())
        for (layer in layers) {
            val w = layer.wList
            val b = layer.bList
            sb.appendLine("${w.rowCount()},${w.columnCount()},${layer.activation.name}")
            for (r in 0 until w.rowCount()) {
                sb.appendLine((0 until w.columnCount()).joinToString(",") { w.getValue(r, it).toString() })
            }
            for (r in 0 until b.rowCount()) {
                sb.appendLine((0 until b.columnCount()).joinToString(",") { b.getValue(r, it).toString() })
            }
        }

        java.io.File(filePath).writeText(sb.toString())
        println("Saved to: ${java.io.File(filePath).absolutePath}")
    }

    companion object {
        /** 从文件导入模型 */
        fun load(filePath: String): NeuralNetwork {
            val lines = java.io.File(filePath).readLines()
            val iter = lines.iterator()
            val layerCount = iter.next().toInt()
            val layers = mutableListOf<NerveNode>()
            for (i in 0 until layerCount) {
                val parts = iter.next().split(",")
                val wRows = parts[0].toInt()
                val wCols = parts[1].toInt()
                val actName = parts[2]
                val w = matrix((0 until wRows).map {
                    iter.next().split(",").map { it.toDouble() }
                })
                val b = matrix((0 until 1).map {
                    iter.next().split(",").map { it.toDouble() }
                })
                val act = requireNotNull(activationMap[actName]) { "Unknown activation: $actName" }
                layers.add(NerveNode(w, b, act))
            }
            return NeuralNetwork(layers)
        }
    }
}