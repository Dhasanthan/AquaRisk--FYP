import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:top_snackbar_flutter/custom_snack_bar.dart';
import 'package:top_snackbar_flutter/top_snack_bar.dart';

import '../../../core/app_export.dart';
import '../models/flood_warning_model.dart';

class FloodWarningController extends GetxController {
  Rx<FloodWarningModel> floodWarningModelObj = FloodWarningModel().obs;

  String district = '';
  String waterLevel = "";
  String rainfall = "";
  bool result = false;
  DateTime dateTime = DateTime.now();

  @override
  void onInit() {
    super.onInit();

    final Map<String, dynamic>? args =
    Get.arguments as Map<String, dynamic>?;

    if (args != null) {
      district = args['district'] ?? '';
      waterLevel = args['waterLevel'] ?? '';
      rainfall = args['rainfall'] ?? '';
      result = args['prediction'] ?? false;
    }
  }

  Future<void> savePrediction(BuildContext context) async {
    try {
      User? user = FirebaseAuth.instance.currentUser;

      if (user != null) {
        String userId = user.uid;

        await FirebaseFirestore.instance.collection('prediction').add({
          'UserId': userId,
          'District': district,
          'Water Level': waterLevel,
          'Rainfall': rainfall,
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
      showTopSnackBar(
        Overlay.of(context),
        CustomSnackBar.error(
          message: "Failed to save Prediction: $e",
        ),
      );
    }
  }

  void onSuccess(BuildContext context) {
    showTopSnackBar(
      Overlay.of(context),
      CustomSnackBar.success(
        message: "Prediction Saved Successfully",
      ),
    );

    Future.delayed(const Duration(seconds: 2), () {
      Get.back();
    });
  }
}