package resnet

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/**
 * What's about Functional API usage in KotlinDL directly?
 *
 * Describe the model like the function of functions, where each layer is just a callable function.
 *
 * Combine two functions in special merge layers like Add or Concatenate.
 *
 * NOTE: Functional API supports one output and one input for model.
 */
fun main() {
    val (train, test) = fashionMnist()

    val inputs = Input(28, 28, 1)
    val conv1 = Conv2D(32)(inputs)
    val conv2 = Conv2D(64)(conv1)
    val maxPool = MaxPool2D(poolSize = intArrayOf(1, 3, 3, 1), strides = intArrayOf(1, 3, 3, 1))(conv2)

    val conv3 = Conv2D(64)(maxPool)
    val conv4 = Conv2D(64)(conv3)
    val add1 = Add()(conv4, maxPool)

    val conv5 = Conv2D(64)(add1)
    val conv6 = Conv2D(64)(conv5)
    val add2 = Add()(conv6, add1)

    val conv7 = Conv2D(64)(add2)
    val globalAvgPool2D = GlobalAvgPool2D()(conv7)
    val dense1 = Dense(256)(globalAvgPool2D)
    val outputs = Dense(10, activation = Activations.Linear)(dense1)

    val model = Functional.fromOutput(outputs)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()

        it.fit(dataset = train, epochs = 3, batchSize = 1000)

        val accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy after: $accuracy")
    }
}
