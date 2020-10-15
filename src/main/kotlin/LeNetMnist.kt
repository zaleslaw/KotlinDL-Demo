
import api.core.Sequential
import api.core.WritingMode
import api.core.activation.Activations
import api.core.initializer.Constant
import api.core.initializer.GlorotNormal
import api.core.initializer.Zeros
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.InferenceModel
import datasets.Dataset
import datasets.handlers.*
import java.io.File


private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

fun main() {
    val model = Sequential.of(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        ),
        Conv2D(
            filters = 32,
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1)
        ),
        Conv2D(
            filters = 64,
            kernelSize = longArrayOf(5, 5),
            strides = longArrayOf(1, 1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME
        ),
        MaxPool2D(
            poolSize = intArrayOf(1, 2, 2, 1),
            strides = intArrayOf(1, 2, 2, 1)
        ),
        Flatten(),
        Dense(
            outputSize = 512,
            activation = Activations.Relu,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Constant(0.1f)
        ),
        Dense(
            outputSize = 10,
            activation = Activations.Linear,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Constant(0.1f)
        )
    )


    val (train, test) = Dataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        10,
        ::extractImages,
        ::extractLabels
    )

    model.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

    model.summary()

    model.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

    model.save(File("/my_model"), writingMode = WritingMode.OVERRIDE)

    val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE)

    println("Accuracy: $accuracy")
}
