import 'package:flutter/material.dart';

class Navbar extends StatelessWidget implements PreferredSizeWidget {
  final int selectedIndex;
  final VoidCallback onDeteksi;
  final VoidCallback onAkun;

  const Navbar({
    super.key,
    required this.selectedIndex,
    required this.onDeteksi,
    required this.onAkun,
  });

  @override
  Size get preferredSize => const Size.fromHeight(50);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      automaticallyImplyLeading: false,
      title: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Expanded(
            child: ElevatedButton(
              onPressed: onDeteksi,
              style: ElevatedButton.styleFrom(
                backgroundColor: selectedIndex == 0 ? Colors.white : const Color(0xFF4285F4),
                foregroundColor: selectedIndex == 0 ? const Color(0xFF4285F4) : Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: const Text('Deteksi', style: TextStyle(fontWeight: FontWeight.bold)),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: ElevatedButton(
              onPressed: onAkun,
              style: ElevatedButton.styleFrom(
                backgroundColor: selectedIndex == 1 ? Colors.white : const Color(0xFF4285F4),
                foregroundColor: selectedIndex == 1 ? const Color(0xFF4285F4) : Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: const Text('Akun', style: TextStyle(fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
      backgroundColor: const Color(0xFF4285F4),
      toolbarHeight: 50,
    );
  }
}
