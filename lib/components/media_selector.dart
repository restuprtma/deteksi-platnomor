import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';


class MediaSelector extends StatelessWidget {
  final File? imageFile;
  final Function(File) onImageSelected;
  final VoidCallback onImageDeleted;

  const MediaSelector({
    super.key,
    required this.imageFile,
    required this.onImageSelected,
    required this.onImageDeleted,
  });

  Future<void> _bukaKamera(BuildContext context) async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);

    if (image != null) {
      final cropped = await _cropImage(image.path);
      if (cropped != null) onImageSelected(cropped);
    }
  }

  Future<void> _bukaGaleri(BuildContext context) async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      final cropped = await _cropImage(image.path);
      if (cropped != null) onImageSelected(cropped);
    }
  }

  Future<File?> _cropImage(String imagePath) async {
    try{
    final croppedFile = await ImageCropper().cropImage(
      sourcePath: imagePath,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Edit Foto',
          toolbarColor: Colors.blue,
          toolbarWidgetColor: Colors.white,
          lockAspectRatio: false,
          hideBottomControls: false,
          cropGridRowCount: 3,
          cropGridColumnCount: 3,
        ),
        IOSUiSettings(
          title: 'Edit Foto',
        ),
      ],
    );

    if (croppedFile == null) return null;
    return File(croppedFile.path);
  }

  catch (e) {
  debugPrint('Error cropping image: $e');
  return null;
  }
}
  @override
  Widget build(BuildContext context) {
    return Container(
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
            height: 200,
            width: double.infinity,
            alignment: Alignment.center,
            child: imageFile != null
                ? Image.file(imageFile!, fit: BoxFit.contain)
                : const Text(
              "Belum ada gambar",
              style: TextStyle(color: Colors.grey),
            ),
          ),

          const Divider(height: 1),

          // Tombol Aksi
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
                  onPressed: () => _bukaKamera(context),
                  icon: const Icon(Icons.camera_alt, color: Colors.white),
                  tooltip: 'Kamera',
                ),
                IconButton(
                  onPressed: () => _bukaGaleri(context),
                  icon: const Icon(Icons.photo_library, color: Colors.white),
                  tooltip: 'Galeri',
                ),
                IconButton(
                  onPressed: onImageDeleted,
                  icon: const Icon(Icons.delete, color: Colors.white),
                  tooltip: 'Hapus',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}