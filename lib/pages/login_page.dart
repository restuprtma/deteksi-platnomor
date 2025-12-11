import 'package:deteksiplat/pages/deteksi_plat_page.dart';
import 'package:flutter/material.dart';


class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  // Variabel input
  String namaPengguna = '';
  String sandi = '';
  bool _obscurePassword = true;

  // Fungsi Login
  void _prosesLogin() {
    // Validasi Hardcode Sederhana
    // Username: admin
    // Sandi: 1234
    if (namaPengguna == 'admin' && sandi == '1234') {
      // Pindah ke DeteksiPlatScreen
      // pushReplacement agar user tidak bisa kembali ke halaman login pakai tombol Back
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const DeteksiPlatPage(),
        ),
      );
    } else {
      // Jika salah password/username
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Username atau Password salah! (Coba: admin / 1234)"),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Text(
          "By Kelompok 3",
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.blueAccent),
        ),
      ),
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("Masuk Sistem",
            style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold
            )
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF4285F4),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 20),
              // Ikon Mobil Besar
              Image.asset('assets/images/plat.png'),
              const SizedBox(height: 20),
              const Text(
                "Deteksi Plat Nomor",
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 40),

              // Input Username
              TextFormField(
                decoration: InputDecoration(
                  labelText: "Nama Pengguna",
                  prefixIcon: const Icon(Icons.person),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                ),
                onChanged: (value) => namaPengguna = value,
              ),

              const SizedBox(height: 20),

              // Input Password
              TextFormField(
                obscureText: _obscurePassword,
                decoration: InputDecoration(
                  labelText: "Sandi",
                  prefixIcon: const Icon(Icons.lock),
                  suffixIcon: IconButton(
                    icon: Icon(_obscurePassword ? Icons.visibility_off : Icons.visibility),
                    onPressed: () {
                      setState(() {
                        _obscurePassword = !_obscurePassword;
                      });
                    },
                  ),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                ),
                onChanged: (value) => sandi = value,
              ),

              const SizedBox(height: 30),

              // Tombol Login
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton(
                  onPressed: _prosesLogin, // Panggil fungsi login di atas
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF4285F4),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: const Text(
                    "Masuk",
                    style: TextStyle(color: Colors.white, fontSize: 18),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}