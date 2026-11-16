import Foundation
import CoreML
import Vision

// JSON Lines keeps the model loaded across requests from the local Python server.
func emit(_ value: [String: Any]) {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([10]))
    } catch {
        FileHandle.standardError.write(Data("JSON encoding failed: \(error)\n".utf8))
    }
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw NSError(domain: "LeafCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: "Provide a Core ML model path."])
    }
    let sourceURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let compiledURL = sourceURL.pathExtension == "mlmodelc"
        ? sourceURL : try MLModel.compileModel(at: sourceURL)
    defer {
        if compiledURL != sourceURL { try? FileManager.default.removeItem(at: compiledURL) }
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
    let visionModel = try VNCoreMLModel(for: model)
    let labels = (model.modelDescription.classLabels ?? []).map { String(describing: $0) }
    emit(["ready": true, "classes": labels, "class_count": labels.count,
          "preprocessing": "EXIF orientation, Vision center crop and scale",
          "name": sourceURL.deletingPathExtension().lastPathComponent])

    while let line = readLine() {
        autoreleasepool {
            do {
                guard let data = line.data(using: .utf8),
                      let input = try JSONSerialization.jsonObject(with: data) as? [String: String],
                      let path = input["image"] else {
                    throw NSError(domain: "LeafCheck", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing image path."])
                }
                let request = VNCoreMLRequest(model: visionModel)
                request.imageCropAndScaleOption = .centerCrop
                let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
                let started = ProcessInfo.processInfo.systemUptime
                try handler.perform([request])
                let elapsed = (ProcessInfo.processInfo.systemUptime - started) * 1000
                guard let observations = request.results as? [VNClassificationObservation],
                      let top = observations.first else {
                    throw NSError(domain: "LeafCheck", code: 3, userInfo: [NSLocalizedDescriptionKey: "Model did not return image classifications."])
                }
                var scores = [String: Double]()
                for item in observations { scores[item.identifier] = Double(item.confidence) }
                emit(["label": top.identifier, "score": Double(top.confidence),
                      "class_scores": scores, "elapsed_ms": elapsed])
            } catch {
                emit(["error": error.localizedDescription])
            }
        }
    }
} catch {
    emit(["error": error.localizedDescription, "ready": false])
    exit(1)
}
