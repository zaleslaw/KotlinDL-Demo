package keras

import LeNetClassic.*
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.handler.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.mnist
import java.io.File

fun main() {
    val (train, test) = mnist()

    val (newTrain, validation) = train.split(0.95)

    lenet5().use {
        it.compile(
            optimizer = SGD(learningRate = 0.05f),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        it.save(
            File("lenet5"),
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")
    }
}

private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val kernelInitializer = GlorotNormal(SEED)
private val biasInitializer = GlorotUniform(SEED)

/**
 * Returns classic LeNet-5 model with minor improvements (Sigmoid activation -> ReLU activation, AvgPool layer -> MaxPool layer).
 */
fun lenet5(): Sequential = Sequential.of(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS,
        name = "input_0"
    ),
    Conv2D(
        filters = 32,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_1"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_2"
    ),
    Conv2D(
        filters = 64,
        kernelSize = longArrayOf(5, 5),
        strides = longArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        padding = ConvPadding.SAME,
        name = "conv2d_3"
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        name = "maxPool_4"
    ),
    Flatten(name = "flatten_5"), // 3136
    Dense(
        outputSize = 120,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_6"
    ),
    Dense(
        outputSize = 84,
        activation = Activations.Relu,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_7"
    ),
    Dense(
        outputSize = NUMBER_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        name = "dense_8"
    )
)