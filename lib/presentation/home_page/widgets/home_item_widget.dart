import '../models/home_item_model.dart';
import '../controller/home_controller.dart';
import 'package:flutter/material.dart';
import 'package:aquarisk/core/app_export.dart';

// ignore: must_be_immutable
class HomeItemWidget extends StatelessWidget {
  HomeItemWidget(
    this.homeItemModelObj, {
    Key? key,
  }) : super(
          key: key,
        );

  HomeItemModel homeItemModelObj;

  var controller = Get.find<HomeController>();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 300.h,
      child: Align(
        alignment: Alignment.centerRight,
        child: SizedBox(
          height: 121.v,
          width: 290.h,
          child: Stack(
            alignment: Alignment.bottomCenter,
            children: [
              Obx(
                () => CustomImageView(
                  imagePath: homeItemModelObj.placeOne!.value,
                  height: 121.v,
                  width: 300.h,
                  radius: BorderRadius.circular(
                    16.h,
                  ),
                  alignment: Alignment.center,
                ),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  padding: EdgeInsets.symmetric(
                    horizontal: 10.h,
                    vertical: 5.v,
                  ),
                  decoration: AppDecoration.gradientBlueGrayToBlueGray.copyWith(
                    borderRadius: BorderRadiusStyle.customBorderBL16,
                  ),
                  // child: Column(
                  //   mainAxisSize: MainAxisSize.min,
                  //   crossAxisAlignment: CrossAxisAlignment.start,
                  //   children: [
                  //     SizedBox(height: 28.v),
                  //     Obx(
                  //       () => Text(
                  //         homeItemModelObj.placeCounter!.value,
                  //         style: CustomTextStyles.titleMediumSourceSansPro,
                  //       ),
                  //     ),
                  //     Obx(
                  //       () => Text(
                  //         homeItemModelObj.address!.value,
                  //         style: CustomTextStyles.bodySmallWhiteA700,
                  //       ),
                  //     ),
                  //   ],
                  // ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
