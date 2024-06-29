import 'package:flutter/material.dart';

import 'package:aquarisk/widgets/custom_drop_down.dart';
import 'package:aquarisk/widgets/custom_text_form_field.dart';
import 'package:aquarisk/widgets/custom_elevated_button.dart';
import 'package:aquarisk/core/app_export.dart';
import 'controller/predict_flood_controller.dart';
import 'models/predict_flood_model.dart';

class PredictFloodPage extends StatefulWidget {
  PredictFloodPage({Key? key}) : super(key: key);

  @override
  State<PredictFloodPage> createState() => _PredictFloodPageState();
}

class _PredictFloodPageState extends State<PredictFloodPage> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  PredictFloodController controller =
  Get.put(PredictFloodController(PredictFloodModel().obs));

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).primaryColor,
          title: Center(
            child: Text(
              "msg_predict_flood_possibility".tr,
              style: TextStyle(color: Colors.white),
            ),
          ),
          automaticallyImplyLeading: false,
        ),
        body: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Form(
            key: _formKey,
            child: Container(
              width: double.maxFinite,
              decoration: AppDecoration.fillWhiteA,
              child: SingleChildScrollView(
                padding: EdgeInsets.symmetric(vertical: 22.v),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildLabel("lbl_district".tr),
                    _buildDropDown(),
                    SizedBox(height: 16.v),
                    _buildLabel("lbl_water_level".tr),
                    _buildMeasurement(),
                    SizedBox(height: 16.v),
                    _buildLabel("lbl_rainfall".tr),
                    _buildInput(),
                    SizedBox(height: 16.v),
                    _buildLabel("lbl_temperature".tr),
                    _buildInput1(),
                    SizedBox(height: 40.v),
                    _buildPredictNow(),
                    SizedBox(height: 5.v),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLabel(String label) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(
        label,
        style: theme.textTheme.titleSmall,
      ),
    );
  }

  Widget _buildDropDown() {
    return CustomDropDown(
      icon: Container(
        margin: EdgeInsets.fromLTRB(30.h, 12.v, 16.h, 12.v),
        child: CustomImageView(
          imagePath: ImageConstant.imgArrowdown,
          height: 24.adaptSize,
          width: 24.adaptSize,
        ),
      ),
      hintText: "Select District",
      items: controller.predictFloodModelObj.value.dropdownItemList.value,
      onChanged: (value) {
        controller.onSelected(value);
      },
    );
  }

  Widget _buildMeasurement() {
    return CustomTextFormField(
      textInputType: TextInputType.number,
      controller: controller.waterLevelController,
      suffix: Text("m"),
      hintStyle: TextStyle(color: Colors.grey),
      hintText: "Eg:3.13".tr,
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter water level';
        }
        // Add additional validation if needed
        return null;
      },
    );
  }

  Widget _buildInput() {
    return CustomTextFormField(
      textInputType: TextInputType.number,
      controller: controller.rainFallController,
      suffix: Text("mm"),
      hintStyle: TextStyle(color: Colors.grey),
      hintText: "Eg:85.12".tr,
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter rainfall';
        }
        // Add additional validation if needed
        return null;
      },
    );
  }

  Widget _buildInput1() {
    return CustomTextFormField(
      textInputType: TextInputType.number,
      controller: controller.temperatureController,
      hintStyle: TextStyle(color: Colors.grey),
      hintText: "Eg:45.15",
      suffix: Text("°C"),
      textInputAction: TextInputAction.done,
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter temperature';
        }
        // Add additional validation if needed
        return null;
      },
    );
  }

  Widget _buildPredictNow() {
    return CustomElevatedButton(
      text: "Predict",
      onPressed: () {
        if (_formKey.currentState!.validate()) {
          controller.predictFlood();
        }
      },
    );
  }
}
