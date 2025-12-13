import 'dart:io';
import 'package:deteksiplat/components/deteksi_plat_proses.dart';
import 'package:deteksiplat/components/media_selector.dart';
import 'package:deteksiplat/components/navbar.dart';
import 'package:deteksiplat/pages/akun_page.dart';
import 'package:flutter/material.dart';

class DeteksiPlatPage extends StatefulWidget {
  const DeteksiPlatPage({super.key});

  @override
  State<DeteksiPlatPage> createState() => _DeteksiPlatPageState();
}

class _DeteksiPlatPageState extends State<DeteksiPlatPage> {
  File? _imageFile;
  String _hasilPlat = "";
  String _akurasiSvm = "";
  String _akurasiCnn = "";
  String _akurasiOverall = "";
  bool _sudahDeteksi = false;
  bool _isLoading = false;
  bool _isError = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: Navbar(
        selectedIndex: 0,
        onDeteksi: () {}, // tetap di halaman ini
        onAkun: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const AkunPage()),
          );
        },
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    blurRadius: 15,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: const [
                      Icon(
                        Icons.info_outline,
                        color: Color(0xFF4285F4),
                        size: 22,
                      ),
                      SizedBox(width: 8),
                      Text(
                        "Cara Kerja",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black,
                        ),
                      ),
                    ],
                  ),

                  SizedBox(height: 10),

                  Text(
                    "Ambil atau unggah foto plat nomor Anda dengan jarak yang pas agar terlihat jelas. "
                    "Sistem kami akan mendeteksi plat nomor dan menampilkan hasilnya beserta akurasi.",
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.4,
                      color: Colors.black87,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 25),

            MediaSelector(
              imageFile: _imageFile,
              onImageSelected: onImageSelected,
              onImageDeleted: onImageDeleted,
            ),

            const SizedBox(height: 30),

            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _prosesDeteksi,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isLoading
                      ? Colors.grey
                      : const Color(0xFF4285F4),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: _isLoading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2.5,
                        ),
                      )
                    : const Text(
                        'Deteksi',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
              ),
            ),

            const SizedBox(height: 30),

            if (_sudahDeteksi) ...[
              Center(
                child: Text(
                  "Hasil klasifikasi:",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[700],
                  ),
                ),
              ),

              const SizedBox(height: 10),

              Container(
                padding: const EdgeInsets.symmetric(
                  vertical: 20,
                  horizontal: 20,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.3),
                      blurRadius: 10,
                      offset: const Offset(0, 5),
                    ),
                  ],
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Flexible(
                      child: Text(
                        _hasilPlat,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: _isError ? Colors.red : Colors.black,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),

                    const SizedBox(width: 10),

                    Icon(
                      _isError ? Icons.error : Icons.check_circle,
                      color: _isError ? Colors.red : Colors.green,
                      size: 28,
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 15),

              Container(
                padding: const EdgeInsets.symmetric(
                  vertical: 20,
                  horizontal: 20,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.3),
                      blurRadius: 10,
                      offset: const Offset(0, 5),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Text(
                        "AKURASI :",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[700],
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    _rowAkurasi("SVM", _akurasiSvm),
                    const SizedBox(height: 8),
                    _rowAkurasi("CNN", _akurasiCnn),
                    const SizedBox(height: 8),
                    _rowAkurasi("Keseluruhan", _akurasiOverall),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  void onImageSelected(File file) {
    setState(() {
      _imageFile = file;
    });
  }

  void onImageDeleted() {
    setState(() {
      _imageFile = null;
    });
  }

  void _prosesDeteksi() async {
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Pilih gambar dulu mas!"),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _isError = false;
      _sudahDeteksi = false;
    });

    try {
      final hasil = await DeteksiPlatProses.prosesDeteksi(_imageFile!);

      final isErrorResult =
          hasil["hasil"] == "Error" ||
          hasil["hasil"] == "Server Error" ||
          hasil["hasil"] == "Koneksi Error";

      setState(() {
        _hasilPlat = hasil["hasil"] ?? "";
        _akurasiSvm = hasil["svm"] ?? "";
        _akurasiCnn = hasil["cnn"] ?? "";
        _akurasiOverall = hasil["overall"] ?? "";
        _sudahDeteksi = true;
        _isError = isErrorResult;
        _isLoading = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              isErrorResult
                  ? "Gagal mendeteksi plat: ${hasil['overall']}"
                  : "Berhasil mendeteksi plat!",
            ),
            backgroundColor: isErrorResult ? Colors.red : Colors.green,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _hasilPlat = "Error";
        _akurasiSvm = "-";
        _akurasiCnn = "-";
        _akurasiOverall = e.toString();
        _sudahDeteksi = true;
        _isError = true;
        _isLoading = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Terjadi kesalahan: $e"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Widget _rowAkurasi(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 16, color: Colors.green)),
        Text(value, style: const TextStyle(fontSize: 16, color: Colors.green)),
      ],
    );
  }
}
