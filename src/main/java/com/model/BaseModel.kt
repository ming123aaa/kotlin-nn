package com.model

import com.ai.NeuralNetwork
import com.ai.matrix


abstract class BaseModel<IN, OUT>(val neuralNetwork: NeuralNetwork) {


    fun run(input: IN): OUT {
        return result(neuralNetwork.forward(inputMatrix(input)))
    }

    fun run(inputs: List<IN>): List<OUT> {
        return neuralNetwork.forward(matrix {
            inputs.forEach {
                row(inputMatrix(it))
            }
        }).data.map {
            result(it)
        }
    }

    fun forward(inputs: List<IN>): List<OUT> {
        return neuralNetwork.forward(matrix {
            inputs.forEach {
                row(inputMatrix(it))
            }
        }).data.map {
            result(it)
        }
    }

    fun train(inputs: List<IN>, outs: List<OUT>, learningRate: Double = 0.1, batchSize: Int = 64) {
        neuralNetwork.trainMiniBatch(
            matrix {
                inputs.forEach { input ->
                    row(inputMatrix(input))
                }

            }, matrix {
                outs.forEach { out ->
                    row(outputMatrix(out))
                }
            }, learningRate, batchSize
        )
    }

    fun save(filePath: String) {
        neuralNetwork.save(filePath)
    }


    abstract fun inputMatrix(input: IN): DoubleArray
    abstract fun outputMatrix(output: OUT): DoubleArray
    abstract fun result(outMatrix: DoubleArray): OUT

}

