package ml.titanic

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.text.NumberFormat
import java.util.*

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 50
private const val TRAINING_BATCH_SIZE = 50

fun main() {
    val df = DataFrame.readCSV(fileOrUrl = "src/main/resources/titanic.csv", delimiter = ';')

    val format = NumberFormat.getInstance(Locale.FRANCE)

    // Calculating imputing values
    val sibspAvg = df["sibsp"].filter { it != null }.map { (it as Int).toDouble() }.values().average()
    val parchAvg = df["parch"].filter { it != null }.map { (it as Int).toDouble() }.values().average()
    val ageAvg = df["age"].filter { it != null }.map { format.parse(it as String).toDouble() }.values().average()
    val fareAvg = df["fare"].filter { it != null }.map { format.parse(it as String).toDouble() }.values().average()

    val (train, test) = df
        .rename("\uFEFFpclass").into("pclass")
        // imputing
        .fillNulls("sibsp").with { sibspAvg }
        .fillNulls("parch").with { parchAvg }
        .fillNulls("age").with { ageAvg.toString() }
        .fillNulls("fare").with { fareAvg.toString() }
        .fillNulls("sex").with { "female" }
        .fillNulls("embarked").with { "S" }
        // conversion
        .convert("age").with { format.parse(it as String).toDouble() }
        .convert("fare").with { format.parse(it as String).toDouble() }
        // one hot encoding
        .oneHotEncoding(columnName = "pclass")
        .oneHotEncoding(columnName = "sex")
        .oneHotEncoding(columnName = "embarked")
        // feature extraction
        .select("survived", "pclass_1", "pclass_2", "pclass_3", "sibsp", "parch", "age", "fare", "sex_1", "sex_2", "embarked_1", "embarked_2", "embarked_3")
        .convert("survived", "pclass_1", "pclass_2", "pclass_3", "sibsp", "parch", "age", "fare", "sex_1", "sex_2", "embarked_1", "embarked_2", "embarked_3")
        .toFloat()
        .shuffled()
        .toOnHeapDataset(labelColumnName = "survived")
        .split(0.7)

    val model = buildNeuralNetwork(numberOfFeatures = 12)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.summary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy: $accuracy")

        // Predict the probability to survive for random person ("Jack Dawson") with the trained model
        // "pclass_1", "pclass_2", "pclass_3", "sibsp", "parch", "age", "fare", "sex_1", "sex_2", "embarked_1", "embarked_2", "embarked_3"
        /*
        If Jack existed in real life, these models predict that he would have died more likely than not and not from hypothermia because he Rose wasn't willing to take turns on that make shift raft.
        Based on the fact that, Jack had
        - Third class tickets.
        - 20 years old male.
        - Didn't have any parents or children with him.
        - Had no siblings with him.
         */
        val jackProfile = floatArrayOf(
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f
        )
        println("Jack will die with probability: ${it.predictSoftly(jackProfile)[0]}")

        // What is about Rose?

        val roseProfile = floatArrayOf(
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 20.0f, 87.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        )
        println("Rose will survive with probability: ${it.predictSoftly(roseProfile)[1]}")
    }
}

private fun buildNeuralNetwork(numberOfFeatures: Int) = Sequential.of(
    Input(numberOfFeatures.toLong()),
    Dense(numberOfFeatures * 5, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeUniform(SEED)),
    Dense(numberOfFeatures * 5, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = HeUniform(SEED)),
    Dense(2, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = HeUniform(SEED))
)

private fun <T> DataFrame<T>.oneHotEncoding(columnName: String, removeSourceColumn: Boolean = true): DataFrame<T> {
    var result = this

    this[columnName].distinct().values().forEachIndexed { index, value ->
        result = result.add(columnName + "_${index + 1}") { if (it[columnName] == value) 1 else 0 }
    }

    if (removeSourceColumn) result = result.remove(columnName)

    return result
}

private fun <T> DataFrame<T>.toOnHeapDataset(labelColumnName: String): OnHeapDataset {
    return OnHeapDataset.create(
        dataframe = this,
        yColumn = labelColumnName
    )
}

private fun OnHeapDataset.Companion.create(
    dataframe: DataFrame<Any?>,
    yColumn: String
): OnHeapDataset {
    fun extractX(): Array<FloatArray> {
        val converted = dataframe.remove(yColumn)
            .map {
                (values() as List<Float>).toFloatArray()
            }.toTypedArray()

        return converted
    }

    fun extractY(): FloatArray {
        val labels = FloatArray(dataframe.nrow()) { 0.0f }
        for (i in labels.indices) {
            val classLabel = dataframe[i][yColumn]
            labels[i] = classLabel as Float
        }

        return labels
    }

    return create(
        ::extractX,
        ::extractY
    )
}





