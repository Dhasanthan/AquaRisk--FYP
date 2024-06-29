import '../../../core/app_export.dart';

/// This class is used in the [home_item_widget] screen.
class HomeItemModel {
  HomeItemModel({
    this.placeOne,
    this.placeCounter,
    this.address,
    this.id,
  }) {
    placeOne = placeOne ?? Rx(ImageConstant.imgRectangle3);
    placeCounter = placeCounter ?? Rx("Place 1");
    address = address ?? Rx("Address");
    id = id ?? Rx("");
  }

  Rx<String>? placeOne;

  Rx<String>? placeCounter;

  Rx<String>? address;

  Rx<String>? id;
}
