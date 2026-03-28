import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../../../core/app_export.dart';

class PredictFloodController extends GetxController {
  final TextEditingController waterLevelController = TextEditingController();
  final TextEditingController rainFallController = TextEditingController();

  final RxString selectedDistrict = 'Matara'.obs;
  final RxList<String> districtList = <String>[
    'Matara',
    'Trincomalee',
    'Kegalle',
    'Polonnaruwa',
    'Kurunegala',
  ].obs;

  final RxString predictionText = ''.obs;
  final RxString probabilityText = ''.obs;
  final RxBool isLoading = false.obs;

  bool result = false;

  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  Map<String, dynamic>? _featureMappings;
  Map<String, dynamic>? _targetMapping;
  Map<String, dynamic>? _scalerParams;

  @override
  void onInit() {
    super.onInit();
    loadModelAndFiles();
  }

  @override
  void onClose() {
    waterLevelController.dispose();
    rainFallController.dispose();
    _interpreter?.close();
    super.onClose();
  }

  Future<void> loadModelAndFiles() async {
    try {
      isLoading.value = true;

      final modelBytes = await rootBundle.load(
        'assets/model/AquaRisk_model.tflite',
      );
      debugPrint('Model found: ${modelBytes.lengthInBytes} bytes');

      _interpreter = await Interpreter.fromAsset(
        'assets/model/AquaRisk_model.tflite',
      );
      debugPrint('Interpreter created successfully');

      final featureMappingsJson = await rootBundle.loadString(
        'assets/model/AquaRiskfeature_mappings.json',
      );
      final targetMappingJson = await rootBundle.loadString(
        'assets/model/AquaRisktarget_mapping.json',
      );
      final scalerParamsJson = await rootBundle.loadString(
        'assets/model/AquaRiskscaler_params.json',
      );

      _featureMappings =
      json.decode(featureMappingsJson) as Map<String, dynamic>;
      _targetMapping = json.decode(targetMappingJson) as Map<String, dynamic>;
      _scalerParams = json.decode(scalerParamsJson) as Map<String, dynamic>;

      _isModelLoaded = true;
      debugPrint('All assets loaded successfully');
    } catch (e) {
      debugPrint('Load error: $e');
      Get.snackbar(
        'Model Load Error',
        e.toString(),
        snackPosition: SnackPosition.BOTTOM,
      );
    } finally {
      isLoading.value = false;
    }
  }

  Future<void> predictFlood() async {
    if (!_isModelLoaded || _interpreter == null) {
      Get.snackbar(
        'Error',
        'Model is not loaded yet.',
        snackPosition: SnackPosition.BOTTOM,
      );
      return;
    }

    try {
      isLoading.value = true;

      final double rainfall = double.parse(rainFallController.text.trim());
      final double riverWaterLevel =
      double.parse(waterLevelController.text.trim());
      final String location = selectedDistrict.value;

      if (location.isEmpty) {
        throw Exception('Please select a district.');
      }

      final locationMap =
      Map<String, dynamic>.from(_featureMappings!['Location'] as Map);

      if (!locationMap.containsKey(location)) {
        throw Exception('District "$location" not found in model mapping.');
      }

      final featureOrder =
      List<String>.from(_scalerParams!['feature_order'] as List);

      final mean = List<double>.from(
        (_scalerParams!['mean'] as List).map((e) => (e as num).toDouble()),
      );

      final scale = List<double>.from(
        (_scalerParams!['scale'] as List).map((e) => (e as num).toDouble()),
      );

      final rawFeatures = <String, double>{
        'Rainfall': rainfall,
        'River_Water_Level': riverWaterLevel,
        'Location': (locationMap[location] as num).toDouble(),
      };

      final inputData = <double>[];
      for (int i = 0; i < featureOrder.length; i++) {
        final featureName = featureOrder[i];
        final rawValue = rawFeatures[featureName];

        if (rawValue == null) {
          throw Exception('Missing feature: $featureName');
        }

        final scaledValue = (rawValue - mean[i]) / scale[i];
        inputData.add(scaledValue);
      }

      debugPrint('Prepared input: $inputData');

      final inputTensor = [inputData];
      final outputTensor = List.generate(1, (_) => List.filled(2, 0.0));

      _interpreter!.run(inputTensor, outputTensor);

      final probs = List<double>.from(outputTensor[0]);
      final predictedIndex = probs[0] > probs[1] ? 0 : 1;
      final predictedLabel =
          _targetMapping?[predictedIndex.toString()]?.toString() ?? '0';

      result = predictedLabel == '1';

      predictionText.value = result ? 'Flood Risk' : 'No Flood Risk';
      probabilityText.value =
      'No Flood: ${(probs[0] * 100).toStringAsFixed(2)}%\n'
          'Flood: ${(probs[1] * 100).toStringAsFixed(2)}%';

      debugPrint('Probabilities: $probs');
      debugPrint('Predicted label: $predictedLabel');
      debugPrint('Result: $result');

      Get.toNamed(
        AppRoutes.floodWarningScreen,
        arguments: {
          'district': location,
          'waterLevel': waterLevelController.text,
          'rainfall': rainFallController.text,
          'prediction': result,
        },
      );
    } catch (e) {
      debugPrint('Prediction error: $e');
      Get.snackbar(
        'Prediction Error',
        e.toString(),
        snackPosition: SnackPosition.BOTTOM,
      );
    } finally {
      isLoading.value = false;
    }
  }
}