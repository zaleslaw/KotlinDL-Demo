/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package onnx

import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.Stroke
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs

/**
 * This examples demonstrates the light-weight inference API with [SSDObjectDetectionModel] on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts rectangles for the detected objects on a few images located in resources.
 * - The detected rectangles related to the objects are drawn on the images used for prediction.
 */
fun main() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.SSD.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)


        for (i in 1..200) {
            val fileName = "%03d".format(i)
            val imageFile = File("cache/datasets/street/$fileName.png")
            val detectedObjects =
                detectionModel.detectObjects(imageFile = imageFile, topK = 50)

            detectedObjects.forEach {
                println("Found ${it.classLabel} with probability ${it.probability}")
            }

            val filteredObjects =
                detectedObjects.filter { it.classLabel == "car" || it.classLabel == "person" || it.classLabel == "bicycle" }

            visualise(fileName, imageFile, filteredObjects)
        }
    }
}

private fun visualise(
    fileName: String,
    imageFile: File,
    detectedObjects: List<DetectedObject>
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(224, 224, 3)
            colorMode = ColorOrder.BGR
        }
        transformImage {
            resize {
                outputWidth = 1200
                outputHeight = 1200
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    drawAndSaveDetectedObjects(fileName, rawImage, ImageShape(1200, 1200, 3), detectedObjects)
}

private fun drawAndSaveDetectedObjects(
    i: String,
    image: FloatArray,
    imageShape: ImageShape,
    detectedObjects: List<DetectedObject>
) {
    val bufferedImage = image.toBufferedImage(imageShape)

    val newGraphics = bufferedImage.createGraphics()
    newGraphics.drawImage(bufferedImage, 0, 0, null)

    detectedObjects.forEach {
        val pixelWidth = 1
        val pixelHeight = 1

        val top = it.yMin * imageShape.height!! * pixelHeight
        val left = it.xMin * imageShape.width!! * pixelWidth
        val bottom = it.yMax * imageShape.height!! * pixelHeight
        val right = it.xMax * imageShape.width!! * pixelWidth
        if (abs(top - bottom) > 400 || abs(right - left) > 400) return@forEach
        // left, bot, right, top

        // y = columnIndex
        // x = rowIndex
        val yRect = bottom
        val xRect = left

        newGraphics as Graphics2D
        val stroke1: Stroke = BasicStroke(4f)
        when (it.classLabel) {
            "person" -> newGraphics.color = Color.RED
            "bicycle" -> newGraphics.color = Color.BLUE
            "car" -> newGraphics.color = Color.GREEN
        }
        newGraphics.stroke = stroke1
        newGraphics.drawRect(xRect.toInt(), yRect.toInt(), (right - left).toInt(), (top - bottom).toInt())
    }

    ImageIO.write(bufferedImage, "png", File("cache/street/$i.png"))
}

