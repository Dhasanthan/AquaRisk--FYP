
import 'package:flutter/material.dart';
import '../../core/app_export.dart';
import 'controller/splash_controller.dart';


// ignore_for_file: must_be_immutable
class SplashScreen extends GetWidget<SplashController> {
  SplashScreen({Key? key})
      : super(
          key: key,
        );


  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        body: SizedBox(
          width: SizeUtils.width,
          child: Center(
            child: Text(" Aqua Risk",style: TextStyle(
              color: Colors.blue,
              fontSize: 30,
              fontWeight: FontWeight.bold,
            ),),
          ),
        ),
      ),
    );
  }
}


