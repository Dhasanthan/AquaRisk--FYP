import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class CustomFloodInfoTile extends StatelessWidget {
  final String district;
  final DateTime date;
  final bool isFlooded;
  final VoidCallback? onTap;

  CustomFloodInfoTile({
    required this.district,
    required this.date,
    required this.isFlooded,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(4.0),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.2), // Shadow color
              spreadRadius: 2, // Spread radius
              blurRadius: 3, // Blur radius
              offset: Offset(0, 3), // Offset from the box
            ),
          ],
        ),
        padding: EdgeInsets.all(8.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: EdgeInsets.symmetric(
                horizontal: 7,
                vertical: 5,
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [

                  Column(
                    children: [
                      Container(
                        //width: 14,
                        margin: EdgeInsets.all(5),

                        child: Text(
                          '${district} District',
                          maxLines: null,
                          textAlign: TextAlign.justify,
                          style: TextStyle(
                              letterSpacing: 0.24,
                              fontSize: 20,
                              color: Colors.blue,
                              fontWeight: FontWeight.bold),
                        ),
                      ),
                      Container(
                        child: Text(
                          'Date: ${_formatDate(date)}',
                          maxLines: null,
                          textAlign: TextAlign.left,
                          style: TextStyle(
                            letterSpacing: 0.24,
                            fontSize: 18,
                            fontStyle: FontStyle.normal,
                            color: Colors.grey,
                          ),
                        ),
                      ),
                    ],
                  ),

                  Padding(
                    padding: const EdgeInsets.only(top: 20.0, bottom: 10.0),
                    child: Container(

                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(10),
                        color: isFlooded ? Colors.green :Colors.redAccent,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.grey.withOpacity(0.2), // Shadow color
                            spreadRadius: 2, // Spread radius
                            blurRadius: 2, // Blur radius
                            offset: Offset(0, 2), // Offset from the box
                          ),
                        ],
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                             '${isFlooded ? 'Flooded' : 'No Flood'}',style: TextStyle(
                          color: Colors.white70
                        ),),
                      )
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
    // return ListTile(
    //   leading: Icon(Icons.location_city),
    //   title: Text(district),
    //   subtitle: Text('Date: ${_formatDate(date)} | ${isFlooded ? 'Flooded' : 'No Flood'}'),
    //   onTap: onTap,
    // );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }
}
