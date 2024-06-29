import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:top_snackbar_flutter/custom_snack_bar.dart';
import 'package:top_snackbar_flutter/top_snack_bar.dart';

import '../../../core/app_export.dart';
import '../models/flood_warning_model.dart';

/// A controller class for the FloodWarningScreen.
///
/// This class manages the state of the FloodWarningScreen, including the
/// current floodWarningModelObj
class FloodWarningController extends GetxController {
  Rx<FloodWarningModel> floodWarningModelObj = FloodWarningModel().obs;

  String district = '';
  String waterLevel = "";
  String rainfall = "";
  String temperature = "";
  bool result = false;
  DateTime dateTime = DateTime.now();

  @override
  void onInit() {
    // TODO: implement onInit
    super.onInit();
    Map<String, dynamic>? args = Get.arguments as Map<String, dynamic>?;

    if (args != null) {
      district = args['district'];
      waterLevel = args['waterLevel'];
      rainfall = args['rainfall'];
      temperature = args['temperature'];
      result = args['prediction'];
    }
  }

  Future<void> savePrediction(BuildContext context) async {
    try {
      // Get current user
      User? user = FirebaseAuth.instance.currentUser;

      if (user != null) {
        String userId = user.uid;

        await FirebaseFirestore.instance.collection('prediction').add({
          'UserId': userId,
          'District': district,
          'Water Level': waterLevel,
          'Rainfall': rainfall,
          'Temperature': temperature,
          'Flood Prediction': result,
          'Date': dateTime,
          'status': 'process',
        });

        onSuccess(context);
      } else {
        throw FirebaseAuthException(
          code: 'user-not-authenticated',
          message: 'User is not authenticated.',
        );
      }
    } catch (e) {
      print(e);
      showTopSnackBar(
        Overlay.of(context),
        CustomSnackBar.error(
          message: "Failed to save Prediction: $e",
        ),
      );
    }
  }

  onSuccess(BuildContext context) {
    showTopSnackBar(
      Overlay.of(context),
      CustomSnackBar.success(
        message: "Prediction Saved Successfully",
      ),
    );
    Future.delayed(Duration(seconds: 2), () {
      Get.back();
    });
  }
}
