import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:top_snackbar_flutter/custom_snack_bar.dart';
import 'package:top_snackbar_flutter/top_snack_bar.dart';

import '../../../core/app_export.dart';
import '../models/district_model.dart';
import 'package:flutter/material.dart';

class SignupController extends GetxController {
  TextEditingController emailController = TextEditingController();
  TextEditingController phoneController = TextEditingController();
  TextEditingController nameController = TextEditingController();

  TextEditingController passwordController = TextEditingController();

  Rx<DistrictModel> districtModelObj = DistrictModel().obs;

  Rx<bool> isShowPassword = true.obs;
  Rx<String> district = "".obs;

  SelectionPopupModel? selectedDropDownValue;

  @override
  void onClose() {
    super.onClose();
    emailController.dispose();
    passwordController.dispose();
    phoneController.dispose();
    nameController.dispose();
  }

  onSelected(SelectionPopupModel? value) {
    for (var element in districtModelObj.value.dropdownItemList.value) {
      element.isSelected = false;
      if (element.id == value!.id) {
        element.isSelected = true;
        district.value = value.title;
      }
    }
    districtModelObj.value.dropdownItemList.refresh();
  }

  void registerUserWithEmailAndPassword(BuildContext context) async {
    try {
      UserCredential userCredential =
          await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: emailController.text,
        password: passwordController.text,
      );

      await FirebaseFirestore.instance
          .collection('users')
          .doc(userCredential.user?.uid)
          .set({
        'Name': nameController.text,
        'email': emailController.text,
        'phoneNumber': phoneController.text,
        'district': district.value
      });
      onSuccess(context);

    } catch (e) {
      showTopSnackBar(
        Overlay.of(context),
        CustomSnackBar.error(
          message: "Failed to create user: $e",
        ),
      );
    }
  }

  void onSuccess(BuildContext context) {
    Get.offNamed(AppRoutes.loginScreen);
    showTopSnackBar(
      Overlay.of(context),
      CustomSnackBar.success(
        message: "User Created Successfully",
      ),
    );
  }
}
