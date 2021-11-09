
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.AdaGradDA
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.text.NumberFormat
import java.util.*

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 50

private val model = Sequential.of(
    Input(5),
    Dense(100, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(30, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(2, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
    val df = DataFrame.readCSV(fileOrUrl = "src/main/resources/titanic.csv", delimiter = ';')
    df.take(10).print() // TODO: fix printing
    println(df.schema())
    df.describe().print() // ?? Bug: should 14 columns not 5 - looks like stat only for numerical columns, it's strange

    val subset = df
        .dropNulls("sibsp", "parch", "age", "fare")
        //.map { it[sex] }
            // TODO: how to make OHE?
        .select("survived", "\uFEFFpclass", "sibsp", "parch", "age", "fare")

    println("------ Subset ------------")
    println(subset.schema())
    subset.describe().print()
    println(subset[0][0])

    df["fare"].describe().print() // TODO: add facts about column like min/max/sum/avg/median/std
    val shuffledDf = subset.shuffled()

    //shuffledDf
    //    .toHeapDataset(labels = "survived")

    // shuffledDf.splitRows(proportion = 0.8)
    val dataset = shuffledDf.toOnHeapDataset(labels = "survived")
    val (train, test) = dataset.split(0.7)

    model.use {
        it.compile(
            optimizer = SGD(learningRate = 0.01f),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

private fun <T> DataFrame<T>.toOnHeapDataset(labels: String): OnHeapDataset {
    return OnHeapDataset.create(
        dataframe = this,
        xColumns = listOf("\uFEFFpclass", "sibsp", "parch", "age", "fare"),
        yColumn = "survived"
    )
}

private fun OnHeapDataset.Companion.create(
    dataframe: DataFrame<Any?>,
    xColumns: List<String>,
    yColumn: String
): OnHeapDataset {
    fun extractX(): Array<FloatArray> {

        val format = NumberFormat.getInstance(Locale.FRANCE) // TODO: parsing strange numbers

        val converted = dataframe.remove(yColumn)
            .convert("\uFEFFpclass", "sibsp", "parch").toFloat()
            .convert("age").with { format.parse(it as String).toFloat() }
            .convert("fare").with { format.parse(it as String).toFloat() }
            .map {
                (values() as List<Float>).toFloatArray()
            }.toTypedArray()

        return converted
    }

    fun extractY(): FloatArray {
        val labels = FloatArray(dataframe.nrow()) { 0.0f }
        for (i in labels.indices) {
            val classLabel = dataframe[i][yColumn]
            labels[i] = (classLabel as Int).toFloat()
        }

        return labels
    }

    return create(
        ::extractX,
        ::extractY
    )
}






