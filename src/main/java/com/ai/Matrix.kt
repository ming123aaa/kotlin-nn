package com.ai

/** 矩阵类，底层使用原生 double[][] 存储，消除 Double 装箱开销 */
class Matrix(val data: Array<DoubleArray>) {

    fun rowCount(): Int = data.size

    fun columnCount(): Int = if (data.isEmpty()) 0 else data[0].size

    /** 获取第 row 行，返回原生 DoubleArray（避免拷贝） */
    fun getRow(row: Int): DoubleArray = data[row]

    /** 获取指定位置的值 */
    fun getValue(row: Int, col: Int): Double = data[row][col]

    /** 设置指定位置的值 */
    fun setValue(row: Int, col: Int, value: Double) {
        data[row][col] = value
    }
}

/** DSL 风格矩阵构建器，用于逐行构建矩阵 */
class MatrixBuilder {
    private val rows = mutableListOf<DoubleArray>()

    /** 从 List<Double> 添加一行 */
    fun row(row: List<Double>) {
        rows.add(row.toDoubleArray())
    }

    /** 从 DoubleArray 添加一行 */
    fun row(row: DoubleArray) {
        rows.add(row)
    }

    fun build(): Matrix = Matrix(rows.toTypedArray())
}

/** DSL 方式构建矩阵，示例: matrix { row(listOf(1.0, 2.0)) } */
fun matrix(builder: MatrixBuilder.() -> Unit): Matrix {
    return MatrixBuilder().apply { builder() }.build()
}

/** 从 List<List<Double>> 构建矩阵 */
fun matrix(data: List<List<Double>>): Matrix {
    return Matrix(Array(data.size) { i -> data[i].toDoubleArray() })
}

/** 从 List<DoubleArray> 构建矩阵（行切片等场景） */
@kotlin.jvm.JvmName("matrixFromDoubleArrays")
fun matrix(data: List<DoubleArray>): Matrix {
    return Matrix(data.toTypedArray())
}

/** 矩阵转置: M×N → N×M */
fun Matrix.transpose(): Matrix {
    val rows = columnCount()
    val cols = rowCount()
    val result = Array(rows) { DoubleArray(cols) }
    val src = data
    for (i in 0 until rows) {
        val row = result[i]
        for (j in 0 until cols) {
            row[j] = src[j][i]
        }
    }
    return Matrix(result)
}

/** 逐元素相乘 (Hadamard 积): C[i][j] = A[i][j] * B[i][j] */
fun Matrix.elementWiseTimes(other: Matrix): Matrix {
    require(rowCount() == other.rowCount() && columnCount() == other.columnCount())
    val rows = rowCount()
    val cols = columnCount()
    val result = Array(rows) { DoubleArray(cols) }
    val a = data
    val b = other.data
    for (i in 0 until rows) {
        val ai = a[i]
        val bi = b[i]
        val ri = result[i]
        for (j in 0 until cols) {
            ri[j] = ai[j] * bi[j]
        }
    }
    return Matrix(result)
}

/** 逐元素相减: C[i][j] = A[i][j] - B[i][j] */
fun Matrix.minus(other: Matrix): Matrix {
    require(rowCount() == other.rowCount() && columnCount() == other.columnCount())
    val rows = rowCount()
    val cols = columnCount()
    val result = Array(rows) { DoubleArray(cols) }
    val a = data
    val b = other.data
    for (i in 0 until rows) {
        val ai = a[i]
        val bi = b[i]
        val ri = result[i]
        for (j in 0 until cols) {
            ri[j] = ai[j] - bi[j]
        }
    }
    return Matrix(result)
}

/** 标量乘法: 每个元素乘以标量 */
operator fun Matrix.times(scalar: Double): Matrix {
    val rows = rowCount()
    val cols = columnCount()
    val result = Array(rows) { DoubleArray(cols) }
    val src = data
    for (i in 0 until rows) {
        val si = src[i]
        val ri = result[i]
        for (j in 0 until cols) {
            ri[j] = si[j] * scalar
        }
    }
    return Matrix(result)
}

/** 标量除法: 每个元素除以标量 */
operator fun Matrix.div(scalar: Double): Matrix {
    val rows = rowCount()
    val cols = columnCount()
    val result = Array(rows) { DoubleArray(cols) }
    val src = data
    for (i in 0 until rows) {
        val si = src[i]
        val ri = result[i]
        for (j in 0 until cols) {
            ri[j] = si[j] / scalar
        }
    }
    return Matrix(result)
}

/** 按列求和: N×M → 1×M，用于偏置梯度计算 */
fun Matrix.sumRows(): Matrix {
    val cols = columnCount()
    val result = DoubleArray(cols)
    val src = data
    val rows = src.size
    for (i in 0 until rows) {
        val row = src[i]
        for (j in 0 until cols) {
            result[j] += row[j]
        }
    }
    return Matrix(arrayOf(result))
}

/** 矩阵乘法: A[M×N] × B[N×P] = C[M×P]
 *  内循环跳过零值，减少 ReLU 之后无效计算 */
operator fun Matrix.times(value: Matrix): Matrix {
    require(columnCount() == value.rowCount()) {
        "Matrix multiply requires left.columns(${columnCount()}) == right.rows(${value.rowCount()})"
    }
    val m = rowCount()
    val n = columnCount()
    val p = value.columnCount()
    val result = Array(m) { DoubleArray(p) }
    val left = data
    val right = value.data
    for (i in 0 until m) {
        val leftRow = left[i]
        val resultRow = result[i]
        for (k in 0 until n) {
            val lk = leftRow[k]
            if (lk != 0.0) {
                val rightRow = right[k]
                for (j in 0 until p) {
                    resultRow[j] += lk * rightRow[j]
                }
            }
        }
    }
    return Matrix(result)
}

/** 矩阵加法: 逐元素相加 */
operator fun Matrix.plus(value: Matrix): Matrix {
    require(rowCount() == value.rowCount() && columnCount() == value.columnCount()) {
        "Matrix add requires same dimensions, got (${rowCount()},${columnCount()}) + (${value.rowCount()},${value.columnCount()})"
    }
    val rows = rowCount()
    val cols = columnCount()
    val result = Array(rows) { DoubleArray(cols) }
    val a = data
    val b = value.data
    for (i in 0 until rows) {
        val ai = a[i]
        val bi = b[i]
        val ri = result[i]
        for (j in 0 until cols) {
            ri[j] = ai[j] + bi[j]
        }
    }
    return Matrix(result)
}