import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';


class DeteksiPlatScreen extends StatefulWidget {
  const DeteksiPlatScreen({super.key});

  @override
  State<DeteksiPlatScreen> createState() => _DeteksiPlatScreenState();
}

class _DeteksiPlatScreenState extends State<DeteksiPlatScreen> {
  File? _imageFile;
  final ImagePicker _picker = ImagePicker();
  String _hasilPlat = "AH 3547 NN";
  String _akurasi = "90 %";
  bool _sudahDeteksi = false;

  Future<void> _bukaKamera() async {
    final XFile? capturedImage = await _picker.pickImage(source: ImageSource.camera);
    if (capturedImage != null) {
      setState(() {
        _imageFile = File(capturedImage.path);
        _sudahDeteksi = false;
      });
    }
  }

  Future<void> _bukaGaleri() async {
    final XFile? pickedImage = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedImage != null) {
      setState(() {
        _imageFile = File(pickedImage.path);
        _sudahDeteksi = false;
      });
    }
  }

  void _hapusGambar() {
    setState(() {
      _imageFile = null;
      _sudahDeteksi = false;
    });
  }

  void _prosesDeteksi() {
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Pilih gambar dulu mas!")),
      );
      return;
    }

    setState(() {
      _sudahDeteksi = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Deteksi Plat Nomor'),
        centerTitle: true,
        backgroundColor: const Color(0xFF4285F4),
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            const Text(
              'Foto Plat Nomor',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.grey,
              ),
            ),
            const SizedBox(height: 10),

            Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.grey.shade300),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    blurRadius: 5,
                    spreadRadius: 1,
                  )
                ],
              ),
              child: Column(
                children: [
                  // Area Gambar
                  Container(
                    height: 200, // Tinggi area gambar
                    width: double.infinity,
                    alignment: Alignment.center,
                    child: _imageFile != null
                        ? Image.file(_imageFile!, fit: BoxFit.contain)
                        : const Text(
                      "Belum ada gambar",
                      style: TextStyle(color: Colors.grey),
                    ),
                  ),

                  const Divider(height: 1),

                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: const BoxDecoration(
                      color: Color(0xFF4285F4),
                      borderRadius: BorderRadius.only(
                        bottomLeft: Radius.circular(10),
                        bottomRight: Radius.circular(10),
                      ),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        IconButton(
                          onPressed: _bukaKamera,
                          icon: const Icon(Icons.camera_alt, color: Colors.white),
                          tooltip: 'Kamera',
                        ),
                        IconButton(
                          onPressed: _bukaGaleri,
                          icon: const Icon(Icons.photo_library, color: Colors.white),
                          tooltip: 'Galeri',
                        ),
                        IconButton(
                          onPressed: _hapusGambar,
                          icon: const Icon(Icons.delete, color: Colors.white),
                          tooltip: 'Hapus',
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 30),

            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton(
                onPressed: _prosesDeteksi,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF4285F4),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text(
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
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 15),
                decoration: BoxDecoration(
                  color: const Color(0xFF4285F4),
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      blurRadius: 5,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    'Akurasi : $_akurasi',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 15),

              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 15),
                decoration: BoxDecoration(
                  color: const Color(0xFF4285F4),
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      blurRadius: 5,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    'Hasil: $_hasilPlat',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}