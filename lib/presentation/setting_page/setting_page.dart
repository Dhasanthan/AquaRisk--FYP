import 'package:aquarisk/core/app_export.dart';
import 'package:aquarisk/presentation/setting_page/controller/setting_controller.dart';
import 'package:flutter/material.dart';

class SettingPage extends StatefulWidget {
  SettingPage({Key? key})
      : super(
          key: key,
        );

  @override
  State<SettingPage> createState() => _SettingPageState();
}

class _SettingPageState extends State<SettingPage> {
  SettingController settingController = Get.put(SettingController());

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).primaryColor,
        title: Center(child: Text("Settings",style: TextStyle(color: Colors.white),)),
        automaticallyImplyLeading: false,
      ),
      body: Obx(() {
        if (settingController.userModel.value == null) {
          return Center(child: CircularProgressIndicator());
        } else {
          //
          return Container(
            width: double.maxFinite,
            padding: EdgeInsets.symmetric(
              horizontal: 10.h,
              vertical: 24.v,
            ),
            child: Column(
              children: [
                Container(
                  height: 122.v,
                  width: 148.h,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(74.h),
                    border: Border.all(color: Colors.green, width: 4.h),
                  ),
                ),
                Container(
                  child: Center(
                    child: Text(
                      settingController.userModel.value!.role != null && settingController.userModel.value!.role!.isNotEmpty
                          ? '${settingController.userModel.value!.name} (${settingController.userModel.value!.role!})'
                          : settingController.userModel.value!.name!,
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 30),
                    ),

                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Container(
                    child: Column(
                      children: [
                        ListTile(
                          title: Text(
                            "Email",
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 22),
                          ),
                          subtitle: Text(
                            settingController.userModel.value!.email.toString(),
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 20),
                          ),
                        ),
                        ListTile(
                          title: Text(
                            "District",
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 22),
                          ),
                          subtitle: Text(
                            settingController.userModel.value!.district.toString(),
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 20),
                          ),
                        ),
                        ListTile(
                          title: Text(
                            "Phone Number",
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 22),
                          ),
                          subtitle: Text(
                            settingController.userModel.value!.phoneNumber.toString(),
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 20),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                ElevatedButton(
                  //style: raisedButtonStyle,
                  onPressed: () {
                    settingController.signOut();
                  },
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: Text('Log Out',style: TextStyle(color: Colors.white,fontSize: 20),),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.redAccent,
                    foregroundColor: Colors.grey[300],
                    minimumSize: Size(88, 36),
                    padding: EdgeInsets.symmetric(horizontal: 16),
                    shape: const RoundedRectangleBorder(
                      borderRadius: BorderRadius.all(Radius.circular(20)),
                    ),
                  ),
                )
              ],
            ),
          );
        }
      }),
    );
  }
}
