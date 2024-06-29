import 'package:aquarisk/widgets/custom_elevated_button.dart';
import 'package:aquarisk/widgets/custom_outlined_button.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:aquarisk/core/app_export.dart';
import 'controller/flood_warning_controller.dart';

// ignore_for_file: must_be_immutable
class FloodWarningScreen extends GetWidget<FloodWarningController> {
  const FloodWarningScreen({Key? key})
      : super(
          key: key,
        );

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        body: Container(
          width: double.maxFinite,
          padding: EdgeInsets.only(
            left: 30.h,
            top: 142.v,
            right: 30.h,
          ),

          child: Column(

            children: [
              if (controller.result == true)
                Column(
                children: [
                  Text(
                    "lbl_flood_warning".tr,
                    style: CustomTextStyles.titleLargeRedA100,
                  ),
                  Text(
                    "msg_there_could_be_a".tr,
                    style: CustomTextStyles.bodySmallPoppinsOnPrimary,
                  ),
                  SizedBox(height: 1.v),
                  Text(
                    "msg_take_necessary_precautions".tr,
                    style: CustomTextStyles.bodySmallPoppinsOnPrimary,
                  ),
                  SizedBox(height: 35.v),
                  CustomElevatedButton(
                    text: "Save",
                    onPressed: (){
                      controller.savePrediction(context);
                    },
                  ),
                  SizedBox(height: 20.v),
                  CustomOutlinedButton(
                    text: "Close",
                    buttonStyle: CustomButtonStyles.outlinePrimary,
                    onPressed: (){
                      Get.back();
                    },
                  ),
                  SizedBox(height: 5.v),
                ],
              )
              else
                Column(
                  children: [
                    Text(
                      "No Flood Warning",
                      style: TextStyle(
                        color: Colors.green
                      ),
                    ),
                    Text(
                      "There is no flood alert in your area.",
                      style: CustomTextStyles.bodySmallPoppinsOnPrimary,
                    ),
                    SizedBox(height: 1.v),

                    SizedBox(height: 35.v),
                    CustomElevatedButton(
                      text: "Save",
                      onPressed: (){
                        controller.savePrediction(context);
                      },
                    ),
                    SizedBox(height: 20.v),
                    CustomOutlinedButton(
                      text: "Close",
                      buttonStyle: CustomButtonStyles.outlinePrimary,
                      onPressed: (){
                        Get.back();
                      },
                    ),
                    SizedBox(height: 5.v),
                  ],
                )
            ],
          ),
        ),

      ),
    );
  }


}
