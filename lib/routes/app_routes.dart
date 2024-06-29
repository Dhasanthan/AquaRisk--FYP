import 'package:aquarisk/presentation/setting_page/setting_page.dart';
import 'package:aquarisk/presentation/splash_screen/binding/splash_binding.dart';
import 'package:get/get.dart';
import '../presentation/home_container_screen/home_container_screen.dart';
import '../presentation/home_container_screen/binding/home_container_binding.dart';
import '../presentation/flood_warning_screen/flood_warning_screen.dart';
import '../presentation/flood_warning_screen/binding/flood_warning_binding.dart';
import '../presentation/app_navigation_screen/app_navigation_screen.dart';
import '../presentation/app_navigation_screen/binding/app_navigation_binding.dart';
import '../presentation/login_screen/binding/login_binding.dart';
import '../presentation/login_screen/login_screen.dart';
import '../presentation/setting_page/binding/setting_binding.dart';
import '../presentation/signup_screen/binding/signup_binding.dart';
import '../presentation/signup_screen/signup_screen.dart';
import '../presentation/splash_screen/splash_screen.dart';

class AppRoutes {
  static const String homeContainerScreen = '/home_container_screen';

  static const String homePage = '/home_page';

  static const String predictFloodPage = '/predict_flood_page';

  static const String floodWarningScreen = '/flood_warning_screen';

  static const String appNavigationScreen = '/app_navigation_screen';

  static const String initialRoute = '/initialRoute';

  static const String loginScreen = '/login_screen';

  static const String splashScreen = '/splash_screen';

  static const String signupScreen = '/signup_screen';

  static const String settingScreen = '/setting_page';

  static List<GetPage> pages = [
    GetPage(
      name: homeContainerScreen,
      page: () => HomeContainerScreen(),
      bindings: [
        HomeContainerBinding(),
      ],
    ),

    GetPage(
      name: loginScreen,
      page: () => LoginScreen(),
      bindings: [
        LoginBinding(),
      ],
    ),
    GetPage(
      name: signupScreen,
      page: () => SignupScreen(),
      bindings: [
        SignupBinding(),
      ],
    ),

    GetPage(
      name: floodWarningScreen,
      page: () => FloodWarningScreen(),
      bindings: [
        FloodWarningBinding(),
      ],
    ),
    GetPage(
      name: appNavigationScreen,
      page: () => AppNavigationScreen(),
      bindings: [
        AppNavigationBinding(),
      ],
    ),
    GetPage(
      name: initialRoute,
      page: () => HomeContainerScreen(),
      bindings: [
        HomeContainerBinding(),
      ],
    ),
    GetPage(
      name: splashScreen,
      page: () => SplashScreen(),
      bindings: [
        SplashBinding(),
      ],
    ),
    GetPage(
      name: settingScreen,
      page: () => SettingPage(),
      bindings: [
        SettingBinding(),
      ],
    )
  ];
}
