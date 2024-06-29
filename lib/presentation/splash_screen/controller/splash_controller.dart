import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';

import '../../../core/app_export.dart';

class SplashController extends GetxController {
  @override
  void onInit() {
    // TODO: implement onInit
    super.onInit();
    Timer(Duration(seconds: 3), () {
      if (FirebaseAuth.instance.currentUser != null) {
        Get.offNamed(AppRoutes.homeContainerScreen);
      } else {
        Get.offNamed(AppRoutes.loginScreen);
      }
    });
  }
}
