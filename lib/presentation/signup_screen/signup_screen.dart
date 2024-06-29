

import 'package:flutter/material.dart';
import 'package:top_snackbar_flutter/custom_snack_bar.dart';
import 'package:top_snackbar_flutter/top_snack_bar.dart';

import '../../core/app_export.dart';
import '../../core/utils/validation_functions.dart';
import '../../widgets/custom_drop_down.dart';
import '../../widgets/custom_elevated_button.dart';
import '../../widgets/custom_text_form_field.dart';
import 'controller/signup_controller.dart';

// ignore_for_file: must_be_immutable
class SignupScreen extends GetWidget<SignupController> {
  SignupScreen({Key? key})
      : super(
          key: key,
        );

  GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        body: SizedBox(
          width: SizeUtils.width,
          child: SingleChildScrollView(
            padding: EdgeInsets.only(
              bottom: MediaQuery.of(context).viewInsets.bottom,
            ),
            child: SizedBox(
              height: SizeUtils.height,
              child: Form(
                key: _formKey,
                child: Container(
                  width: double.maxFinite,
                  padding: EdgeInsets.symmetric(
                    horizontal: 38.h,
                    vertical: 36.v,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(height: 22.v),
                      Align(
                        alignment: Alignment.center,
                        child: Text(
                          "lbl_signup".tr,
                          style: theme.textTheme.titleMedium,
                        ),
                      ),
                      SizedBox(height: 20.v),
                      Text(
                        "lbl_name".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 6.v),
                      CustomTextFormField(
                        controller: controller.nameController,
                        hintText: "lbl_example_name".tr,
                        textInputType: TextInputType.text,
                        validator: (value) {
                          if (!isValidName(value)) {
                            return "err_msg_please_enter_name".tr;
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 15.v),
                      Text(
                        "lbl_email".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 6.v),
                      CustomTextFormField(
                        controller: controller.emailController,
                        hintText: "lbl_example_email".tr,
                        textInputType: TextInputType.emailAddress,
                        validator: (value) {
                          if (!isValidGmail(value)) {
                            return "err_msg_please_enter_valid_email".tr;
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 15.v),
                      Text(
                        "lbl_number".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 6.v),
                      CustomTextFormField(
                        controller: controller.phoneController,
                        hintText: "lbl_07712345678".tr,
                        textInputType: TextInputType.phone,
                        validator: (value) {
                          if (!isValidPhone(value)) {
                            return "err_msg_please_enter_valid_phone_number".tr;
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 15.v),
                      Text(
                        "lbl_district".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 8.v),
                      CustomDropDown(
                        icon: Container(
                          margin: EdgeInsets.fromLTRB(30.h, 12.v, 16.h, 12.v),
                          child: CustomImageView(
                            imagePath: ImageConstant.imgArrowdown,
                            height: 24.adaptSize,
                            width: 24.adaptSize,
                          ),
                        ),
                        hintText: "Select District",
                        items: controller
                            .districtModelObj.value.dropdownItemList.value,
                        onChanged: controller.onSelected,
                      ),
                      SizedBox(height: 15.v),
                      Text(
                        "lbl_password".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 8.v),
                      Obx(
                        () => CustomTextFormField(
                          controller: controller.passwordController,
                          textInputAction: TextInputAction.done,
                          suffix: InkWell(
                            onTap: () {
                              controller.isShowPassword.value =
                                  !controller.isShowPassword.value;
                            },
                            child: Container(
                              margin:
                                  EdgeInsets.fromLTRB(30.h, 12.v, 16.h, 12.v),
                              child: CustomImageView(
                                imagePath: ImageConstant.imgIconEye,
                                height: 24.adaptSize,
                                width: 24.adaptSize,
                              ),
                            ),
                          ),
                          suffixConstraints: BoxConstraints(
                            maxHeight: 48.v,
                          ),
                          obscureText: controller.isShowPassword.value,
                        ),
                      ),
                      SizedBox(height: 40.v),
                      CustomElevatedButton(
                          text: "lbl_sign_up2".tr,
                          onPressed: () {
                            if (_formKey.currentState!.validate()) {
                              controller.registerUserWithEmailAndPassword(context);
                            } else {
                              showTopSnackBar(
                                Overlay.of(context),
                                CustomSnackBar.error(
                                  message:
                                      "Please correct the errors in the form.",
                                ),
                              );
                            }
                          }),
                      Spacer(),
                      Align(
                        alignment: Alignment.center,
                        child: Text(
                          "msg_already_have_an".tr,
                          style: theme.textTheme.bodySmall,
                        ),
                      ),
                      SizedBox(height: 4.v),
                      Align(
                        alignment: Alignment.center,
                        child: GestureDetector(
                          onTap: () => onTabSignIn(),
                          child: Text(
                            "lbl_login".tr,
                            style: CustomTextStyles.bodyMediumPrimary.copyWith(
                              decoration: TextDecoration.underline,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

onTabSignIn() {
  Get.toNamed(AppRoutes.loginScreen);
}
