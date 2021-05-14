package resnet

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.model.resnet50Light
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
    /*val (train, test) = fashionMnist()

    val model = resnet50Light(imageSize = 28, numberOfClasses = 10, numberOfChannels = 1, lastLayerActivation = Activations.Linear)

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
    }*/
}
