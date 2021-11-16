/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package onnx

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.awt.*
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFrame
import javax.swing.JPanel
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

            visualise(
                fileName,
                imageFile,
                detectedObjects.filter { it.classLabel == "car" || it.classLabel == "person" || it.classLabel == "bicycle" })
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

    val frame = JFrame("Filters")
    @Suppress("UNCHECKED_CAST")
    frame.contentPane.add(DetectedObjectJPanel(bufferedImage, imageShape, detectedObjects))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
    ImageIO.write(bufferedImage, "png", File("cache/street/$i.png"))
}

private class DetectedObjectJPanel(
    private val bufferedImage: BufferedImage,
    private val imageShape: ImageShape,
    private val detectedObjects: List<DetectedObject>
) : JPanel() {
    override fun paintComponent(g: Graphics?) {
        super.paintComponent(g)
    }

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val newGraphics = bufferedImage.createGraphics()
        newGraphics.drawImage(bufferedImage, 0, 0, null)

        detectedObjects.forEach {
            val pixelWidth = 1
            val pixelHeight = 1

            val top = it.yMin * imageShape.height!! * pixelHeight
            val left = it.xMin * imageShape.width!! * pixelWidth
            val bottom = it.yMax * imageShape.height!! * pixelHeight
            val right = it.xMax * imageShape.width!! * pixelWidth
            if (abs(top - bottom) > 400 || abs(right - left) > 400) return
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
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

fun FloatArray.toBufferedImage(imageShape: ImageShape): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), BufferedImage.TYPE_INT_RGB)
    for (i in 0 until imageShape.height!!.toInt()) { // rows
        for (j in 0 until imageShape.width!!.toInt()) { // columns
            val r = get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            val g = get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            val b = get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            result.setRGB(j, i, Color(r, g, b).rgb)
        }
    }
    return result
}

