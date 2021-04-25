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
    var x = Conv2D(32)(inputs)
    x = Conv2D(64)(x)
    val block1Output = MaxPool2D(poolSize = intArrayOf(1, 3, 3, 1), strides = intArrayOf(1, 3, 3, 1))(x)

    x = Conv2D(64)(block1Output)
    x = Conv2D(64)(x)
    val block2Output = Add()(x, block1Output)

    x = Conv2D(64)(block2Output)
    x = Conv2D(64)(x)
    val block3Output = Add()(x, block2Output)

    x = Conv2D(64)(block3Output)
    x = GlobalAvgPool2D()(x)
    x = Dense(256)(x)
    val outputs = Dense(10, activation = Activations.Linear)(x)

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
