import 'dart:io';

class DeteksiPlatProses {
  // simulasi proses deteksi plat
  static Future<Map<String, String>> prosesDeteksi(File image) async {
    await Future.delayed(const Duration(seconds: 1));

    // Return data hasil
    return {
      "hasil": "N 1345 HA",
      "svm": "90%",
      "cnn": "85%",
      "overall": "92%",
    };
  }
}
