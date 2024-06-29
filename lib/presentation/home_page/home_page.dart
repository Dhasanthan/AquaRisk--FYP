import 'package:aquarisk/presentation/home_page/widgets/list_item_widget.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'widgets/home_item_widget.dart';
import 'models/home_item_model.dart';

import 'package:flutter/material.dart';
import 'package:aquarisk/core/app_export.dart';
import 'controller/home_controller.dart';
import 'models/home_model.dart';

class HomePage extends StatelessWidget {
  HomePage({Key? key})
      : super(
          key: key,
        );

  HomeController controller = Get.put(HomeController(HomeModel().obs));
  final Stream<QuerySnapshot> _predictionStream = FirebaseFirestore.instance.collection('prediction').snapshots();
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        appBar: AppBar(
          backgroundColor: Theme.of(context).primaryColor,
          title: Center(child: Text("Welcome",style: TextStyle(color: Colors.white),)),
          automaticallyImplyLeading: false,
        ),
        body: Container(
          width: double.maxFinite,
          decoration: AppDecoration.fillWhiteA,
          child: SingleChildScrollView(
            child: Column(
              children: [

                SizedBox(height: 7.v),
                _buildHome(),
                SizedBox(height: 10.v),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Padding(
                    padding: EdgeInsets.only(left: 29.h),
                    child: Column(
                      children: [
                        Text(
                          "msg_previous_predications".tr,
                          style: CustomTextStyles.titleMediumOnPrimary18,
                        ),
                        SizedBox(height: 8),
                        Container(
                          width: 200.h, // Example width, adjust as needed
                          height: 2,
                          color: Colors.blue, // Example primary color
                        ),

                      ],
                    ),

                  ),
                ),

                _buildPredictionList()

              ],
            ),
          ),
        ),
      ),
    );
  }

  /// Section Widget
  Widget _buildHome() {
    return Align(
      alignment: Alignment.centerRight,
      child: SizedBox(
        height: 150.h,
        child: Obx(() {
          final homeItems = controller.homeModelObj.value.homeItemList.value;
          return CarouselSlider(
            options: CarouselOptions(
              height: 150,
              autoPlay: true,
              autoPlayInterval: Duration(seconds: 5),
              onPageChanged: (index, reason) {
                controller.currentIndex.value = index;
              },
            ),
            items: homeItems.map((item) {
              return Builder(
                builder: (BuildContext context) {
                  return HomeItemWidget(item);
                },
              );
            }).toList(),
          );
        }),
      ),
    );
  }

  Widget _buildPredictionList() {
    return StreamBuilder<QuerySnapshot>(
      stream: _predictionStream,
      builder: (BuildContext context, AsyncSnapshot<QuerySnapshot> snapshot) {
        if (snapshot.hasError) {
          return Center(
            child: Text('Error: ${snapshot.error}'),
          );
        }

        if (snapshot.connectionState == ConnectionState.waiting) {
          return Center(
            child: CircularProgressIndicator(),
          );
        }

        List<DocumentSnapshot> predictions = snapshot.data!.docs;

        return ListView.builder(
          shrinkWrap: true,
          physics: NeverScrollableScrollPhysics(),
          itemCount: predictions.length,
          itemBuilder: (context, index) {
            Map<String, dynamic> data = predictions[index].data() as Map<String, dynamic>;
            String district = data['District'] ?? '';
            DateTime date = (data['Date'] as Timestamp).toDate();
            bool isFlooded = data['Flood Prediction'] ?? false;

            // Return the CustomFloodInfoTile widget using extracted data
            return Padding(
              padding: const EdgeInsets.only(left: 10.0,right: 10,top: 10),
              child: CustomFloodInfoTile(
                district: district,
                date: date,
                isFlooded: isFlooded,
              ),
            );
          },
        );
      },
    );
  }

}
