import 'package:aquarisk/presentation/home_page/home_page.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../../core/app_export.dart';
import '../../login_screen/login_screen.dart';
import '../models/UserModel.dart';


class SettingController extends GetxController {

  FirebaseAuth _auth = FirebaseAuth.instance;
  FirebaseFirestore _firestore = FirebaseFirestore.instance;


  Rx<UserModel?> userModel = Rx<UserModel?>(null);

  @override
  void onInit() {
    super.onInit();
    getUserData();

  }



  Future<void> getUserData() async {
    final user = FirebaseAuth.instance.currentUser;

    if (user != null) {
      DocumentSnapshot userData =
      await _firestore.collection('users').doc(user.uid).get();
      userModel.value = UserModel.fromMap(userData.data() as Map<String, dynamic>);
      userModel.value!.uid = user.uid;

    }
  }

  Future<void> signOut() async {
    await _auth.signOut();
    Get.offAllNamed(AppRoutes.loginScreen);
  }


  @override
  void onClose() {
    super.onClose();

  }
}
