import 'package:aquarisk/core/app_export.dart';
import '../../../core/app_export.dart';

class PredictFloodModel {
  Rx<List<SelectionPopupModel>> dropdownItemList = Rx([
    SelectionPopupModel(
      id: 1,
      title: "Ratnapura",
      isSelected: true,
    ),
    SelectionPopupModel(
      id: 2,
      title: "Kalutara",
    ),

  ]);
}
