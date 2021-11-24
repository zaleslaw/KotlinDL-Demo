import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.features.*
import io.ktor.client.request.forms.*
import io.ktor.client.statement.*
import io.ktor.http.*
import kotlinx.coroutines.runBlocking
import resnet.getFileFromResource
import java.io.File

fun main() {
    runBlocking {
        val client = HttpClient(CIO)

        val response: HttpResponse = client.submitFormWithBinaryData(
            url = "http://localhost:8000/upload",
            formData = formData {
                append("description", "Ktor logo")
                append("image", getFileFromResource("datasets/vgg/image9.jpg").readBytes(), Headers.build {
                    append(HttpHeaders.ContentType, "image/jpg")
                    append(HttpHeaders.ContentDisposition, "filename=image9.jpg")
                })
            }
        ) {
            onUpload { bytesSentTotal, contentLength ->
                println("Sent $bytesSentTotal bytes from $contentLength")
            }
        }

        println(response.readText())
    }
}