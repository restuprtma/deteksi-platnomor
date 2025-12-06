import 'package:flutter/material.dart';
import 'screens/deteksi_plat_screen.dart';

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
      home: const DeteksiPlatScreen(),
    );
  }
}