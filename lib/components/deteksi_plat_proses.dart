import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

class DeteksiPlatProses {
  // - Emulator Android: 10.0.2.2
  // - HP asli: IP WiFi komputer (contoh: 192.168.1.100)
  static const String baseUrl = "http://192.168.1.5:8000";

  static Future<Map<String, String>> prosesDeteksi(File image) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );

      request.files.add(await http.MultipartFile.fromPath('file', image.path));

      var streamedResponse = await request.send().timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          throw Exception('Request timeout - server tidak merespon');
        },
      );

      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = json.decode(response.body);

        if (data['success'] == true) {
          return {
            "hasil": data['hasil'] ?? "",
            "svm": data['svm']['accuracy'] ?? "",
            "cnn": data['cnn']['accuracy'] ?? "",
            "overall": data['overall'] ?? "",
          };
        } else {
          return {
            "hasil": "Error",
            "svm": "-",
            "cnn": "-",
            "overall": data['error'] ?? "Unknown error",
          };
        }
      } else {
        return {
          "hasil": "Server Error",
          "svm": "-",
          "cnn": "-",
          "overall": "Status: ${response.statusCode}",
        };
      }
    } catch (e) {
      return {
        "hasil": "Koneksi Error",
        "svm": "-",
        "cnn": "-",
        "overall": e.toString(),
      };
    }
  }

  // Method untuk cek status server
  static Future<bool> checkServerHealth() async {
    try {
      var response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        return data['status'] == 'ok';
      }
      return false;
    } catch (e) {
      return false;
    }
  }
}
