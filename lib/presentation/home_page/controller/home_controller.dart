import '../../../core/app_export.dart';
import '../models/home_model.dart';
import 'package:flutter/material.dart';

/// A controller class for the HomePage.
///
/// This class manages the state of the HomePage, including the
/// current homeModelObj
class HomeController extends GetxController {
  HomeController(this.homeModelObj);

  TextEditingController predictionCounterController = TextEditingController();

  TextEditingController predictionCounterController1 = TextEditingController();

  TextEditingController predictionCounterController2 = TextEditingController();

  TextEditingController predictionCounterController3 = TextEditingController();

  Rx<HomeModel> homeModelObj;

  var currentIndex = 0.obs;


  void updateImage() {
    currentIndex.value = (currentIndex.value + 1) % homeModelObj.value.homeItemList.value.length;
  }

  @override
  void onClose() {
    super.onClose();
    predictionCounterController.dispose();
    predictionCounterController1.dispose();
    predictionCounterController2.dispose();
    predictionCounterController3.dispose();
  }
}
