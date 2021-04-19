/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package resnet

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotUniform
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelZoo
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.catDogsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 8
private const val TEST_BATCH_SIZE = 16
private const val NUM_CLASSES = 2
private const val NUM_CHANNELS = 3L
private const val IMAGE_SIZE = 224L
private const val TRAIN_TEST_SPLIT_RATIO = 0.7

fun resnet50additionalTraining() {
    val modelZoo =
        ModelZoo(commonModelDirectory = File("cache/pretrainedModels"), modelType = ModelType.ResNet_50)
    val model = modelZoo.loadModel() as Functional

    val catdogimages = catDogsSmallDatasetPath()

    val preprocessing: Preprocessing = preprocessingPipeline {
        imagePreprocessing {
            load {
                pathToData = File(catdogimages)
                imageShape = ImageShape(channels = NUM_CHANNELS)
                colorMode = ColorOrder.BGR
                labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
            }
            resize {
                outputHeight = IMAGE_SIZE.toInt()
                outputWidth = IMAGE_SIZE.toInt()
                interpolation = InterpolationType.BILINEAR
            }
        }
        sharpen {
            modelType = ModelType.ResNet_50
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing).shuffle()
    val (train, test) = dataset.split(TRAIN_TEST_SPLIT_RATIO)

    val hdfFile = modelZoo.loadWeights()
    val layers = mutableListOf<Layer>()

    for (layer in model.layers) {
        layer.isTrainable = false
        layers.add(layer)
    }

    val lastLayer = layers.last()
    for (outboundLayer in lastLayer.inboundLayers)
        outboundLayer.outboundLayers.remove(lastLayer)

    layers.removeLast()

    var x = Dense(
        name = "top_dense",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = 200,
        activation = Activations.Relu
    )(layers.last())

    x = Dense(
        name = "pred",
        kernelInitializer = GlorotUniform(),
        biasInitializer = GlorotUniform(),
        outputSize = NUM_CLASSES,
        activation = Activations.Linear
    )(x)

    val model2 = Functional.fromOutput(x)

    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.loadWeightsForFrozenLayers(hdfFile)
        val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBeforeTraining")

        it.fit(
            dataset = train,
            batchSize = TRAINING_BATCH_SIZE,
            epochs = EPOCHS
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}

fun main() = resnet50additionalTraining()


