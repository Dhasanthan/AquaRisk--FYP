import 'package:aquarisk/presentation/flood_warning_screen/controller/flood_warning_controller.dart';
import 'package:get/get.dart';

/// A binding class for the FloodWarningScreen.
///
/// This class ensures that the FloodWarningController is created when the
/// FloodWarningScreen is first loaded.
class FloodWarningBinding extends Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => FloodWarningController());
  }
}
