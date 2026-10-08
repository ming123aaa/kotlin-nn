# AI 神经网络示例代码

从零实现的神经网络框架（Kotlin），包含矩阵运算库、全连接层、多种激活函数、训练算法，以及一个 BMI 预测演示应用。

## 项目结构

```
src/main/
├── kotlin/
│   └── Main.kt                    # 入口（含 HTTP GET 示例）
└── java/com/
    ├── ai/
    │   ├── Matrix.kt              # 矩阵运算库（乘法、转置、逐元素运算等）
    │   └── NerveNode.kt           # 神经网络核心（层、激活函数、初始化器、训练）
    ├── model/
    │   ├── BaseModel.kt           # 领域模型抽象基类
    │   └── BimModel.kt            # BMI 预测应用（生成数据、训练、评估）
    └── util/
        ├── MD5.java               # MD5 哈希工具
        ├── HttpGet.java           # HTTP GET 客户端
        ├── FileUtils.java         # 文件读写工具
        └── CopyFileUtil.kt        # 文件复制与逐行转换工具
```

## 功能特性

### 神经网络框架 (`com.ai`)

| 模块 | 说明 |
|------|------|
| **矩阵运算** | 矩阵乘法（含 ReLU 零值跳过优化）、转置、加减、逐元素乘、标量乘除、按行求和 |
| **层 (NerveNode)** | 全连接层 (Dense)，支持前向传播、反向传播、梯度下降更新 |
| **激活函数** | Sigmoid、ReLU、Softmax、Linear（均含前向和导数计算） |
| **权重初始化** | He 初始化（适用于 ReLU）、Xavier 初始化（适用于 Sigmoid/Tanh） |
| **网络 (NeuralNetwork)** | 多层串联，支持单样本 SGD、全批量、Mini-batch 三种训练模式 |
| **模型序列化** | `save()` / `load()` 将权重保存为纯文本文件 |

### BMI 预测应用 (`com.model.BimModel`)

- 输入：身高 (1.50~1.90m) 和体重 (40~110kg)，归一化处理
- 输出：BMI / 100（回归值）
- 网络结构：2 输入 → 8 隐藏层 (各 128 神经元, ReLU) → 1 输出 (Sigmoid)
- 两阶段训练：先单样本 SGD 10 轮，再 Mini-batch SGD 10 轮，学习率逐步衰减
- 自动生成合成数据集进行训练和验证

## 环境要求

- JDK 8+
- Gradle 7.4.2（项目自带 Gradle Wrapper，无需手动安装）

## 构建与运行

```bash
# 构建项目
./gradlew build          # Linux / macOS
gradlew.bat build        # Windows

# 运行 BMI 模型训练（在 IntelliJ IDEA 中直接运行 BimModel.kt 的 main 方法）
# 或通过命令行：
./gradlew jar
java -cp build/libs/test-1.0-SNAPSHOT.jar com.model.BimModelKt

# 运行测试
./gradlew test

# 清理构建产物
./gradlew clean
```

## 架构概览

```text
BimModel (BaseModel<IN, OUT>)
  └── NeuralNetwork
        └── List<NerveNode>
              ├── W (权重矩阵)
              ├── b (偏置向量)
              └── Activation (Sigmoid / ReLU / Softmax / Linear)
```

`BaseModel` 提供泛型训练接口，子类只需实现 `inputMatrix()`、`outputMatrix()` 和 `result()` 即可将神经网络应用于特定领域问题。

## 依赖

- **Kotlin 1.7.20** — 主要开发语言
- **Gson 2.9.0** — JSON 解析（预留依赖）
- 无第三方机器学习库——所有神经网络运算均为纯 Kotlin/Java 实现