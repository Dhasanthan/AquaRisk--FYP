import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import '../../../core/app_export.dart';
import '../models/predict_flood_model.dart';
import 'package:flutter/material.dart';

class PredictFloodController extends GetxController {
  PredictFloodController(this.predictFloodModelObj);

  TextEditingController waterLevelController = TextEditingController();

  TextEditingController rainFallController = TextEditingController();

  TextEditingController temperatureController = TextEditingController();

  DateTime dateTime = DateTime.timestamp();

  String district = "";

  Rx<PredictFloodModel> predictFloodModelObj;

  SelectionPopupModel? selectedDropDownValue;

  bool result = false;

  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  @override
  void onInit() {
    // TODO: implement onInit
    super.onInit();
    _loadModel();
  }

  @override
  void onClose() {
    super.onClose();
    waterLevelController.clear();
    rainFallController.clear();
    temperatureController.clear();
  }

  onSelected(SelectionPopupModel value) {
    for (var element in predictFloodModelObj.value.dropdownItemList.value) {
      element.isSelected = false;
      if (element.id == value.id) {
        element.isSelected = true;
        district = element.title;
        print(district);
      }
    }
    predictFloodModelObj.value.dropdownItemList.refresh();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
          'assets/model/neural_network_model.tflite');

      _isModelLoaded = true;
    } catch (e) {
      print('Failed to load TensorFlow Lite model: $e');
    }
  }

  double _convertMetersToFeet(double meters) {
    return meters * 3.28084;
  }

  double _convertMillimetersToInches(double millimeters) {
    return millimeters * 0.0393701;
  }

  void predictFlood() {
    if (!_isModelLoaded) {
      print('TensorFlow Lite model is not loaded.');
      return;
    }

    try {
      double waterLevel = double.parse(waterLevelController.text);
      double rainFall = double.parse(rainFallController.text);
      double temperature = double.parse(temperatureController.text);

      double waterLevelInFeet = _convertMetersToFeet(waterLevel);
      double rainFallInInches = _convertMillimetersToInches(rainFall);

      List<double> inputData = [
        waterLevelInFeet,
        rainFallInInches,
        temperature
      ];

      if (district == 'Ratnapura') {
        inputData.add(1);
      } else if (district == 'Kalutara') {
        inputData.add(2);
      } else {
        inputData.add(0);
      }

      List<double> scaledInputData = _scaleInputData(inputData);

      var inputTensor = Float32List.fromList(scaledInputData);

      var outputTensor = List.filled(1, 0).reshape([1, 1]);

      _interpreter!.run(inputTensor, outputTensor);

      double prediction = outputTensor[0][0];

      result = prediction >= 0.9 ? true : false;
      print(prediction);


      Get.toNamed(AppRoutes.floodWarningScreen, arguments: {
        'district': district,
        'waterLevel': waterLevelController.text,
        'rainfall': rainFallController.text,
        'temperature': temperatureController.text,
        'prediction': result,
      });
    } catch (e) {
      print('Error performing prediction: $e');
    }
  }

  List<double> _scaleInputData(List<double> inputData) {
    double meanWaterLevel = 40.0;
    double stdDevWaterLevel = 10.0;
    double meanRainFall = 20.0;
    double stdDevRainFall = 5.0;
    double meanTemperature = 5.0;
    double stdDevTemperature = 2.0;

    List<double> scaledData = [
      (inputData[0] - meanWaterLevel) / stdDevWaterLevel,
      (inputData[1] - meanRainFall) / stdDevRainFall,
      (inputData[2] - meanTemperature) / stdDevTemperature,
    ];

    return scaledData;
  }
}
