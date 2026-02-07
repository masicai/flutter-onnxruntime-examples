import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img; // Using image package for image processing
import 'dart:math' as math;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classification Demo',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue)),
      home: const OnnxModelDemoPage(title: 'Image Classification Demo'),
    );
  }
}

class OnnxModelDemoPage extends StatefulWidget {
  const OnnxModelDemoPage({super.key, required this.title});

  final String title;

  @override
  State<OnnxModelDemoPage> createState() => _OnnxModelDemoPageState();
}

class _OnnxModelDemoPageState extends State<OnnxModelDemoPage> {
  bool _isProcessing = false;
  OrtSession? _session;
  final assetPath = 'assets/models/resnet18-v1-7.onnx';
  final classNamesPath = 'assets/models/imagenet-simple-labels.json';
  final imagePath = 'assets/images/cat.jpg';
  List<Map<String, dynamic>> _displayResults = [];
  List<OrtProvider> _availableProviders = [];
  String? _selectedProvider;
  // Cache for decoded image to avoid decoding it multiple times
  img.Image? _cachedImage;
  // Cache model size to avoid loading the entire model file every inference
  int? _modelSizeBytes;

  @override
  void initState() {
    super.initState();
    _getModelInfo();
    _loadAndCacheImage();
  }

  // Load and cache the image during initialization
  Future<void> _loadAndCacheImage() async {
    final ByteData imageData = await rootBundle.load(imagePath);
    _cachedImage = img.decodeImage(imageData.buffer.asUint8List());
    if (_cachedImage == null) {
      throw Exception('Failed to decode image');
    }
  }

  Future<void> _getModelInfo() async {
    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath);

    // optional: get and set the execution provider
    _availableProviders = await OnnxRuntime().getAvailableProviders();
    setState(() {
      _selectedProvider = _availableProviders.isNotEmpty ? _availableProviders[0].name : null;
    });

    final modelMetadata = await _session!.getMetadata();
    final modelMetadataMap = modelMetadata.toMap();
    final List<Map<String, dynamic>> modelInputInfoMap = await _session!.getInputInfo();
    final List<Map<String, dynamic>> modelOutputInfoMap = await _session!.getOutputInfo();

    // generate a list of maps from the modelMetadataMap if the values is not empty
    final displayList = [
      {'title': 'Model Name', 'value': assetPath.split('/').last},
    ];
    // loop through the list of input and output info and add them to the displayList
    // Add index to the input and output prefix
    for (var i = 0; i < modelInputInfoMap.length; i++) {
      for (var key in modelInputInfoMap[i].keys) {
        displayList.add({'title': 'Input $i: $key', 'value': modelInputInfoMap[i][key].toString()});
      }
    }
    for (var i = 0; i < modelOutputInfoMap.length; i++) {
      for (var key in modelOutputInfoMap[i].keys) {
        displayList.add({'title': 'Output $i: $key', 'value': modelOutputInfoMap[i][key].toString()});
      }
    }

    for (var entry in modelMetadataMap.entries) {
      // if the value is string and empty, skip
      if (entry.value is String && entry.value.isEmpty) {
        continue;
      }
      displayList.add({'title': entry.key, 'value': entry.value.toString()});
    }

    setState(() {
      _displayResults = displayList;
    });
  }

  // Placeholder method to run inference
  Future<void> _runInference() async {
    setState(() {
      _isProcessing = true;
    });

    OrtProvider provider;
    if (_selectedProvider == null) {
      provider = OrtProvider.CPU;
    } else {
      provider = OrtProvider.values.firstWhere((p) => p.name == _selectedProvider);
    }

    final sessionOptions = OrtSessionOptions(providers: [provider]);

    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath, options: sessionOptions);

    // Use the cached image or load it if not available
    if (_cachedImage == null) {
      await _loadAndCacheImage();
    }

    // Use the cached image for processing
    final img.Image image = _cachedImage!;

    // Preprocess image for ResNet model (resize to 224x224 and normalize)
    final img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

    // Convert to RGB float tensor [1, 3, 224, 224] with values normalized between 0-1
    // ResNet models typically expect RGB format with normalization using ImageNet stats
    final Float32List inputData = Float32List(1 * 3 * 224 * 224);

    int pixelIndex = 0;
    for (int c = 0; c < 3; c++) {
      // RGB channels
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          // Get R, G, B values (0-255)
          double value;
          if (c == 0) {
            value = resizedImage.getPixel(x, y).r.toDouble(); // R
          } else if (c == 1) {
            value = resizedImage.getPixel(x, y).g.toDouble(); // G
          } else {
            value = resizedImage.getPixel(x, y).b.toDouble(); // B
          }

          // Normalize to 0-1 range
          value = value / 255.0;

          // Apply ImageNet normalization
          final means = [0.485, 0.456, 0.406];
          final stds = [0.229, 0.224, 0.225];
          value = (value - means[c]) / stds[c];

          inputData[pixelIndex++] = value;
        }
      }
    }

    // Create OrtValue from preprocessed image
    OrtValue inputTensor = await OrtValue.fromList(
      inputData,
      [1, 3, 224, 224], // Input shape: batch, channels, height, width
    );

    // Load class names
    final String classNamesJson = await rootBundle.loadString(classNamesPath);
    final List<dynamic> classNames = jsonDecode(classNamesJson);
    // RestNet18 has only one input and one output so we just get the first one in the lists
    final String inputName = _session!.inputNames.first;
    final String outputName = _session!.outputNames.first;

    // Run inference
    final startTime = DateTime.now();
    final outputs = await _session!.run({
      inputName: inputTensor, // 'data' is the input name for ResNet18
    });
    final endTime = DateTime.now();

    // Get the results
    // Resnet18 returns a float32 list, we cast it to a list of doubles since Dart doesn't support float32
    final List<double> scores = (await outputs[outputName]!.asFlattenedList()).cast<double>();

    // Output from classification models are logits so we have to apply softmax to convert logits to probabilities
    final List<double> probabilities = _applySoftmax(scores);

    // Find top prediction
    int maxIndex = 0;
    double maxProbability = probabilities[0];

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProbability) {
        maxIndex = i;
        maxProbability = probabilities[i];
      }
    }

    // Calculate inference time
    final inferenceTime = endTime.difference(startTime).inMilliseconds;

    // Get the model file size (cached to avoid loading the entire file every time)
    if (_modelSizeBytes == null) {
      final asset = await rootBundle.load(assetPath);
      _modelSizeBytes = asset.lengthInBytes;
    }
    final modelSizeInMB = (_modelSizeBytes! / (1024 * 1024)).toStringAsFixed(1);

    // Clean up resources
    await inputTensor.dispose();
    for (var output in outputs.values) {
      await output.dispose();
    }

    // Update results
    setState(() {
      _displayResults = [
        {'title': 'Model Name', 'value': assetPath.split('/').last},
        {'title': 'Model Size', 'value': '$modelSizeInMB MB'},
        {'title': 'Top Prediction', 'value': '${classNames[maxIndex]} (id: $maxIndex)'},
        {'title': 'Confidence', 'value': maxProbability.toStringAsFixed(4)},
        {'title': 'Inference Time', 'value': '$inferenceTime ms'},
        {'title': 'Processing Device', 'value': _selectedProvider ?? 'CPU'},
      ];
      _isProcessing = false;
    });
  }

  List<double> _applySoftmax(List<double> logits) {
    // Find the maximum value to avoid numerical instability
    double maxLogit = logits.reduce((curr, next) => curr > next ? curr : next);

    // Subtract max from each value for numerical stability
    List<double> expValues = logits.map((logit) => math.exp(logit - maxLogit)).toList();

    // Calculate sum of all exp values
    double sumExp = expValues.reduce((sum, val) => sum + val);

    // Normalize by dividing each by the sum
    return expValues.map((expVal) => expVal / sumExp).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Theme.of(context).colorScheme.inversePrimary, title: Text(widget.title)),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Cat image at the top
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 20.0),
              child: Image.asset('assets/images/cat.jpg', height: 200, fit: BoxFit.contain),
            ),
            // Dropdown for selecting execution provider
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  const Text('Provider:', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                  const SizedBox(width: 20),
                  DropdownButton<String>(
                    value: _selectedProvider,
                    hint: const Text('Select Execution Provider'),
                    items:
                        _availableProviders.map((provider) {
                          return DropdownMenuItem<String>(value: provider.name, child: Text(provider.name));
                        }).toList(),
                    onChanged: (value) {
                      setState(() {
                        _selectedProvider = value;
                      });
                    },
                  ),
                ],
              ),
            ),
            // Get Model Info and Predict buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(onPressed: _getModelInfo, child: const Text('Get Model Info')),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _isProcessing ? null : _runInference,
                  style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 12)),
                  child:
                      _isProcessing
                          ? const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.0)),
                              SizedBox(width: 12),
                              Text('Processing...'),
                            ],
                          )
                          : const Text('Predict', style: TextStyle(fontSize: 16)),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Results section
            Expanded(
              child:
                  _displayResults.isEmpty
                      ? const Center(
                        child: Text(
                          'Press the Predict button to run inference',
                          style: TextStyle(fontSize: 16, color: Colors.grey),
                        ),
                      )
                      : ListView.builder(
                        itemCount: _displayResults.length,
                        itemBuilder: (context, index) {
                          final result = _displayResults[index];
                          return Card(
                            margin: const EdgeInsets.only(bottom: 8),
                            child: Padding(
                              padding: const EdgeInsets.all(12.0),
                              child: Row(
                                children: [
                                  Expanded(
                                    flex: 2,
                                    child: Text(
                                      result['title'],
                                      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                                    ),
                                  ),
                                  Expanded(flex: 3, child: Text(result['value'], style: const TextStyle(fontSize: 14))),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
            ),
          ],
        ),
      ),
    );
  }
}
