import 'package:firebase_auth/firebase_auth.dart';
import 'package:top_snackbar_flutter/custom_snack_bar.dart';
import 'package:top_snackbar_flutter/top_snack_bar.dart';

import '../../../core/app_export.dart';
import '../models/login_model.dart';
import 'package:flutter/material.dart';

class LoginController extends GetxController {
  TextEditingController emailController = TextEditingController();

  TextEditingController passwordController = TextEditingController();

  Rx<LoginModel> loginModelObj = LoginModel().obs;

  Rx<bool> isShowPassword = true.obs;

  late AnimationController localAnimationController;

  @override
  void onClose() {
    super.onClose();
    emailController.dispose();
    passwordController.dispose();
  }

  Future<void> loginUserWithEmailAndPassword(BuildContext context) async {
    try {
      final credential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: emailController.text,
        password: passwordController.text,
      );
      await onSuccess(context);
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        showTopSnackBar(
          Overlay.of(context),
          CustomSnackBar.error(
            message: "No user found for that email.",
          ),
        );
      } else if (e.code == 'wrong-password') {
        showTopSnackBar(
          Overlay.of(context),
          CustomSnackBar.error(
            message: "Wrong password provided for that user.",
          ),
        );
      } else {
        showTopSnackBar(
          Overlay.of(context),
          CustomSnackBar.error(
            message: "Error: ${e.message}",
          ),
        );
      }
    } catch (e) {
      showTopSnackBar(
        Overlay.of(context),
        CustomSnackBar.error(
          message: "Unexpected error: $e",
        ),
      );
    }
  }

  Future<void> onSuccess(BuildContext context) async {
    Get.offNamed(AppRoutes.homeContainerScreen);
    showTopSnackBar(
      Overlay.of(context),
      CustomSnackBar.success(
        message: "Good job, Login successful.",
      ),
    );
  }
}
