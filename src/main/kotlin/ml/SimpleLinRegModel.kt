import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import kotlin.math.exp
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 20

private val model = Sequential.of(
    Input(4),
    Dense(1, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
    val rnd = Random(SEED)
    val data = Array(10000) { doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0) }
    for (i in data.indices) {
        data[i][1] = 2 * (rnd.nextDouble() - 0.5)
        data[i][2] = 2 * (rnd.nextDouble() - 0.5)
        data[i][3] = 2 * (rnd.nextDouble() - 0.5)
        data[i][4] = 2 * (rnd.nextDouble() - 0.5)
        data[i][0] = data[i][1] - 2 * data[i][2] + 1.5 * data[i][3] - 0.95 * data[i][4] + 0.2 + rnd.nextDouble(0.1)
        // 1 * x1 - 2 * x2 + 1.5 * x3  - 0.95 * x4 +- 0.1
    }

    data.shuffle()

    fun extractX(): Array<FloatArray> {
        val init: (index: Int) -> FloatArray = { index ->
            floatArrayOf(
                data[index][1].toFloat(),
                data[index][2].toFloat(),
                data[index][3].toFloat(),
                data[index][4].toFloat()
            )
        }
        return Array(data.size, init = init)
    }

    fun extractY(): FloatArray {
        val labels = FloatArray(data.size) { 0.0f }
        for (i in labels.indices) {
            labels[i] = data[i][0].toFloat()
        }

        return labels
    }

    val dataset = OnHeapDataset.create(
        ::extractX,
        ::extractY
    )

    val (train, test) = dataset.split(0.9)

    model.use {
        it.compile(
            optimizer = SGD(learningRate = 0.001f),
            loss = Losses.MAE,
            metric = Metrics.MAE
        )

        it.summary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val mae = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MAE]
        println("Weights: " + it.getLayer("dense_2").weights["dense_2_dense_kernel"].contentDeepToString())
        println("Bias: " + it.getLayer("dense_2").weights["dense_2_dense_bias"].contentDeepToString())
        println("MAE: $mae")


        repeat(100) { id ->
            val xReal = test.getX(id)
            val yReal = test.getY(id)

            val yPred2 = it.predict(xReal) // always returns 0
            val yPred3 = it.predictSoftly(xReal) // returns value oscillating around 1.0

            println("xReal: ${xReal[0]}, yReal: $yReal, yPred2: $yPred2, yPred3: ${yPred3[0]}")
        }
    }
}


