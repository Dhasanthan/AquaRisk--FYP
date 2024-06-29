class UserModel {
  String? uid;
  String? name;
  String? district;
  String? email;
  String? phoneNumber;
  String? role;

  UserModel({this.uid, this.name, this.district, this.email, this.phoneNumber, this.role});

  factory UserModel.fromMap(Map<String, dynamic> data) {
    return UserModel(
      uid: data['uid'],
      name: data['Name'],
      district: data['district'],
      email: data['email'],
      phoneNumber: data['phoneNumber'],
      role: data['role']
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'name': name,
      'district': district,
      'email': email,
      'phoneNumber': phoneNumber,
      'role' : role
    };
  }
}
