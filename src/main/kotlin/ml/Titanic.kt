
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
    Dense(50, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(2, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
    val df = DataFrame.readCSV(fileOrUrl = "src/main/resources/titanic.csv", delimiter = ';')
    df.take(10).print() // TODO: fix printing
    println(df.schema())
    df.describe().print() // ?? Bug: should 14 columns not 5 - looks like stat only for numerical columns, it's strange

    val subset = df
        .dropNulls { cols(it["sibsp"], it["parch"], it["age"], it["fare"]) }
        //.map { it[sex] }
            // TODO: how to make OHE?
        .select("survived", "\uFEFFpclass", "sibsp", "parch", "age", "fare")

    println("------ Subset ------------")
    println(subset.schema())
    subset.describe().print()
    println(subset[0][0])


    df["fare"].describe().print() // TODO: add facts about column like min/max/sum/avg/median/std
    subset.shuffled()

    val dataset = OnHeapDataset.create(dataframe = subset, xColumns = listOf("\uFEFFpclass", "sibsp", "parch", "age", "fare"), yColumn = "survived")
    val (train, test) = dataset.split(0.9)

    model.use {
        it.compile(optimizer = SGD(learningRate = 0.01f), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.summary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

private fun OnHeapDataset.Companion.create(dataframe: DataFrame<Any?>, xColumns: List<String>, yColumn: String): OnHeapDataset {
    fun extractX(): Array<FloatArray> {

        val format = NumberFormat.getInstance(Locale.FRANCE) // TODO: parsing strange numbers

        val init: (index: Int) -> FloatArray = { index ->
            floatArrayOf(
                (dataframe[index]["\uFEFFpclass"] as Int).toFloat(),
                (dataframe[index]["sibsp"] as Int).toFloat(),
                (dataframe[index]["parch"] as Int).toFloat(),
                format.parse(dataframe[index]["age"] as String).toFloat()/100, // TODO: build statistics and apply custom normalization
                format.parse(dataframe[index]["fare"] as String).toFloat()/1000 // TODO: build statistics and apply custom normalization
            )
        }
        val array = Array(dataframe.nrow(), init = init)
        return array
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






