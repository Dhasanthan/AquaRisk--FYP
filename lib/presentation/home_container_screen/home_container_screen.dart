import 'package:aquarisk/presentation/home_page/home_page.dart';
import 'package:aquarisk/presentation/predict_flood_page/predict_flood_page.dart';
import 'package:aquarisk/presentation/setting_page/setting_page.dart';
import 'package:aquarisk/widgets/custom_bottom_bar.dart';
import 'package:flutter/material.dart';
import 'package:aquarisk/core/app_export.dart';
import 'controller/home_container_controller.dart';

class HomeContainerScreen extends GetWidget<HomeContainerController> {
  const HomeContainerScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SafeArea(
        child: Scaffold(
            body: Navigator(
                key: Get.nestedKey(1),
                initialRoute: AppRoutes.homePage,
                onGenerateRoute: (routeSetting) => GetPageRoute(
                    page: () => getCurrentPage(routeSetting.name!),
                    transition: Transition.noTransition)),
            bottomNavigationBar: Padding(
                padding: EdgeInsets.only(left: 10.h, right: 10.h),
                child: _buildBottomBar())));
  }

  /// Section Widget
  Widget _buildBottomBar() {
    return CustomBottomBar(onChanged: (BottomBarEnum type) {
      Get.toNamed(getCurrentRoute(type), id: 1);
    });
  }

  ///Handling route based on bottom click actions
  String getCurrentRoute(BottomBarEnum type) {
    switch (type) {
      case BottomBarEnum.Home:
        return AppRoutes.homePage;
      case BottomBarEnum.Prediction:
        return AppRoutes.predictFloodPage;
      case BottomBarEnum.Notification:
        return "/";
      case BottomBarEnum.Settings:
        return AppRoutes.settingScreen;
      default:
        return "/";
    }
  }

  ///Handling page based on route
  Widget getCurrentPage(String currentRoute) {
    switch (currentRoute) {
      case AppRoutes.homePage:
        return HomePage();
      case AppRoutes.predictFloodPage:
        return PredictFloodPage();
      case AppRoutes.settingScreen:
        return SettingPage();
      default:
        return DefaultWidget();
    }
  }
}
