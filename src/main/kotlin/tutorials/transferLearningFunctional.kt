package tutorials

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

fun main() {
    val dogsVsCatsDatasetPath = dogsCatsSmallDatasetPath()

   /* val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = File(dogsVsCatsDatasetPath)
            imageShape = ImageShape(channels = 3)
            colorMode = ColorOrder.BGR
            labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
        }
        transformImage {
            resize {
                outputHeight = 224
                outputWidth = 224
                interpolation = InterpolationType.BILINEAR
            }
        }
        transformTensor {
            sharpen {
                TFModels.CV.ResNet50
            }
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
    val (train, test) = dataset.split(0.7)

    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.ResNet50
    val model = modelHub.loadModel(modelType)

    val layers = mutableListOf<Layer>()

    for (layer in model.layers) {
        layer.isTrainable = false
        layers.add(layer)
    }

    val lastLayer = layers.last()
    for (outboundLayer in lastLayer.inboundLayers)
        outboundLayer.outboundLayers.remove(lastLayer)

    layers.removeLast()

    val newDenseLayer = Dense(
        name = "new_dense_1",
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        outputSize = 64,
        activation = Activations.Relu
    )
    newDenseLayer.inboundLayers.add(layers.last())
    layers.add(
        newDenseLayer
    )

    val newDenseLayer2 = Dense(
        name = "new_dense_2",
        kernelInitializer = HeNormal(),
        biasInitializer = HeNormal(),
        outputSize = 2,
        activation = Activations.Linear
    )
    newDenseLayer2.inboundLayers.add(layers.last())

    layers.add(
        newDenseLayer2
    )

    val newModel = Functional.of(layers)

    newModel.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        val hdfFile = modelHub.loadWeights(modelType)
        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = 8,
            epochs = 2
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }*/
}
