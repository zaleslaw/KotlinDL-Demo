package tutorial

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import java.io.File

val labelsMap = mapOf(
    0 to "airplane",
    1 to "automobile",
    2 to "bird",
    3 to "cat",
    4 to "deer",
    5 to "dog",
    6 to "frog",
    7 to "horse",
    8 to "ship",
    9 to "truck"
)

val imageArray = ImageConverter.toNormalizedFloatArray(File("src/resources/models/keras"))

fun main() {
    val modelConfig = File("PATH_TO_MODEL_JSON")
    val weights = File("PATH_TO_WEIGHTS")

    val model = Sequential.loadModelConfiguration(modelConfig)

    model.use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        it.loadWeights(HdfFile(weights))

        val prediction = it.predict(imageArray)
        println("Predicted label is: $prediction. This corresponds to class ${labelsMap[prediction]}.")
    }

    fun main() {
        val (train, test) = fashionMnist()
        InferenceModel.load(File("PATH_TO_MODEL")).use {
            it.reshape(28, 28, 1)
            val prediction = it.predict(test.getX(0))
            val actualLabel = test.getY(0)

            println("Predicted label is: $prediction. This corresponds to class ${labelsMap[prediction]}.")
            println("Actual label is: $actualLabel.")
        }
    }
}
