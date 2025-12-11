import 'package:deteksiplat/pages/login_page.dart';
import 'package:flutter/material.dart';
import 'pages/deteksi_plat_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deteksi Plat',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3B82F6),
          primary: const Color(0xFF3B82F6),
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      home: const LoginPage()
    );
  }
}