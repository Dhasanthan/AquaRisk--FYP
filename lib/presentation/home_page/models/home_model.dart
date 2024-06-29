import '../../../core/app_export.dart';
import 'home_item_model.dart';

/// This class defines the variables used in the [home_page],
/// and is typically used to hold data that is passed between different parts of the application.
class HomeModel {
  Rx<List<HomeItemModel>> homeItemList = Rx([
    HomeItemModel(
        placeOne: ImageConstant.flood1.obs,
        placeCounter: "Place 1".obs,
        address: "Address".obs),
    HomeItemModel(
        placeOne: ImageConstant.flood2.obs,
        placeCounter: "Place 2".obs,
        address: "Address".obs),
    HomeItemModel(
        placeOne: ImageConstant.flood3.obs,
        placeCounter: "Place 1".obs,
        address: "Address".obs),
    HomeItemModel(
        placeOne: ImageConstant.flood4.obs,
        placeCounter: "Place 2".obs,
        address: "Address".obs),
  ]);
}
