
import 'package:aquarisk/core/app_export.dart';
import 'package:flutter/material.dart';
import '../../core/utils/validation_functions.dart';
import '../../widgets/custom_elevated_button.dart';
import '../../widgets/custom_text_form_field.dart';
import 'controller/login_controller.dart';

// ignore_for_file: must_be_immutable
class LoginScreen extends GetWidget<LoginController> {
  LoginScreen({Key? key})
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
                          "lbl_login".tr,
                          style: theme.textTheme.titleMedium,
                        ),
                      ),
                      SizedBox(height: 40.v),
                      Text(
                        "lbl_email".tr,
                        style: theme.textTheme.titleSmall,
                      ),
                      SizedBox(height: 6.v),
                      CustomTextFormField(
                        controller: controller.emailController,
                        hintText: "lbl_example_email".tr,
                        hintStyle: CustomTextStyles.bodyMediumInter,
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
                        onPressed: ()  async {
                        await controller.loginUserWithEmailAndPassword(context);
                        },
                        text: "lbl_login".tr,
                      ),
                      Spacer(),
                      Align(
                        alignment: Alignment.center,
                        child: Text(
                          "msg_don_t_have_an_account".tr,
                          style: theme.textTheme.bodySmall,
                        ),
                      ),
                      SizedBox(height: 5.v),
                      Align(
                        alignment: Alignment.center,
                        child: GestureDetector(
                          onTap: ()=>onTabSignUp(),
                          child: Text(
                            "lbl_signup".tr,
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

onTabSignUp() {
  Get.toNamed(AppRoutes.signupScreen);
}
