"""
Generate synthetic JSONL training data for the Gratifikasi binary classifier.

Labels:
  - Milik Negara     : gratification that must be reported to the state
  - Bukan Milik Negara: gratification that does not belong to the state

Usage:
    python scripts/generate_synthetic_data.py \
        --output data/synthetic_training_data.jsonl \
        --samples 500 \
        --seed 42
"""
import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template pools
# ---------------------------------------------------------------------------

# --- Milik Negara -----------------------------------------------------------
MILIK_NEGARA_TEMPLATES = [
    # Vendor / contractor
    "Pejabat pengadaan menerima {gift} senilai Rp {amount} dari {vendor} yang tengah mengikuti proses tender pengadaan {item} di instansi tersebut.",
    "{vendor} memberikan {gift} senilai Rp {amount} kepada kepala divisi pengadaan sesaat setelah kontrak pengadaan {item} ditandatangani.",
    "Direktur keuangan menerima {gift} dari kontraktor pemenang tender proyek {project} senilai kurang lebih Rp {amount}.",
    "Seorang account manager dari {vendor} menyerahkan {gift} senilai Rp {amount} kepada pejabat berwenang sehari sebelum pengumuman pemenang lelang.",
    "PNS yang menjabat sebagai panitia lelang menerima {gift} senilai Rp {amount} dari salah satu peserta tender.",
    "Kepala bidang perizinan menerima uang tunai Rp {amount} dari {vendor} yang sedang mengurus perpanjangan izin operasional.",
    "{vendor} mentransfer dana Rp {amount} ke rekening pribadi pejabat sebagai ucapan terima kasih atas kelancaran proses pengadaan.",
    "Pejabat eselon II menerima {gift} seharga Rp {amount} dari pimpinan {vendor} pada acara penutupan proyek pengadaan {item}.",
    "Staf pengadaan menerima voucher belanja senilai Rp {amount} dari distributor {item} yang baru saja memenangkan tender.",
    "Inspektur lapangan menerima {gift} senilai Rp {amount} dari rekanan yang proyeknya sedang diperiksa kesesuaian spesifikasi teknisnya.",

    # Tender / procurement
    "Komite evaluasi tender mendapatkan {gift} dari {vendor} senilai Rp {amount} saat rapat pleno penetapan pemenang lelang.",
    "Pejabat yang terlibat dalam penilaian dokumen penawaran menerima {gift} senilai Rp {amount} dari salah satu peserta.",
    "Setelah proses seleksi konsultan selesai, konsultan terpilih memberikan {gift} Rp {amount} kepada ketua panitia seleksi.",
    "Anggota tim negosiasi harga menerima {gift} senilai Rp {amount} dari vendor setelah kesepakatan harga tercapai.",
    "Pejabat pembuat komitmen mendapatkan {gift} senilai Rp {amount} setelah menandatangani kontrak dengan {vendor}.",

    # Seminar / training sponsored by external party
    "Perusahaan {vendor} menanggung seluruh biaya perjalanan dan akomodasi senilai Rp {amount} untuk pejabat yang diundang ke seminar di {city}.",
    "Sebuah vendor farmasi menanggung biaya registrasi seminar Rp {amount} serta hotel bintang lima untuk dokter PNS yang menjadi peserta.",
    "Perusahaan rekanan membiayai keikutsertaan pejabat pada konferensi internasional senilai Rp {amount} tanpa persetujuan instansi.",
    "Sponsor dari {vendor} menanggung biaya perjalanan pejabat ke {city} untuk workshop industri senilai Rp {amount}.",
    "PNS menerima sponsorship perjalanan dinas ke luar negeri senilai Rp {amount} dari perusahaan yang produknya sedang dalam proses evaluasi tender.",

    # Travel / hospitality
    "Rekanan perusahaan menanggung biaya menginap dan transportasi pejabat selama kunjungan kerja ke {city} senilai Rp {amount}.",
    "Pejabat menerima tiket pesawat pulang-pergi kelas bisnis senilai Rp {amount} dari mitra bisnis yang sedang bermitra dengan instansinya.",
    "Vendor membayarkan seluruh tagihan restoran mewah senilai Rp {amount} untuk pejabat dan keluarganya dalam rangka memperkuat hubungan bisnis.",
    "Kontraktor proyek infrastruktur menanggung biaya wisata pejabat dan keluarganya ke Bali senilai Rp {amount}.",
    "Perusahaan jasa konsultasi menanggung biaya akomodasi pejabat beserta keluarga selama tiga malam senilai Rp {amount} di resort bintang lima.",

    # Government-context subtle cases
    "Seorang pengusaha memberikan {gift} senilai Rp {amount} kepada pejabat pajak yang menangani pemeriksaan perpajakannya.",
    "Wajib pajak badan menyerahkan {gift} senilai Rp {amount} kepada petugas pemeriksa pajak setelah proses pemeriksaan dinyatakan selesai.",
    "Seorang notaris memberikan uang tunai Rp {amount} kepada pejabat pertanahan yang memproses akta tanah kliennya.",
    "Pengembang properti memberikan {gift} senilai Rp {amount} kepada pejabat dinas tata ruang yang menerbitkan IMB proyek mereka.",
    "Importir barang mewah memberikan {gift} senilai Rp {amount} kepada petugas bea cukai yang memeriksa barang kirimannya.",

    # Partner / business relationship
    "Mitra strategis perusahaan BUMN memberikan {gift} senilai Rp {amount} kepada direksi BUMN yang menandatangani perjanjian kerja sama.",
    "Perusahaan swasta memberikan saham senilai Rp {amount} kepada pejabat BUMN sebagai imbalan atas dukungan dalam negosiasi kontrak.",
    "Pejabat instansi pemerintah menerima fee konsultasi Rp {amount} dari perusahaan yang kebijakan internalnya sedang diproses oleh instansi tersebut.",
    "Distributor tunggal memberikan {gift} senilai Rp {amount} kepada pejabat yang berwenang memperpanjang hak distribusi eksklusifnya.",
    "Agen perjalanan memberikan cashback Rp {amount} kepada pejabat yang memilih agen tersebut sebagai penyedia layanan perjalanan dinas instansi.",

    # Subtle / complex cases
    "Pejabat menerima {gift} senilai Rp {amount} dari seorang kenalan lama yang ternyata merupakan pemilik perusahaan yang sedang mengajukan izin ke instansinya.",
    "Mantan rekanan memberikan {gift} senilai Rp {amount} kepada pejabat, namun perusahaan rekanan tersebut kini mengajukan proposal kerja sama baru.",
    "Seorang pengusaha yang anaknya pernah dibantu urusan administrasinya memberikan {gift} senilai Rp {amount} kepada pejabat yang bersangkutan.",
    "Pejabat menerima {gift} senilai Rp {amount} dari asosiasi industri yang anggotanya sedang dalam proses evaluasi regulasi oleh instansinya.",
    "Vendor yang belum memenangkan tender memberikan {gift} senilai Rp {amount} kepada pejabat pengadaan sebagai \"ucapan perkenalan\".",
    "Perusahaan minyak dan gas memberikan {gift} senilai Rp {amount} kepada pejabat kementerian yang mengeluarkan izin eksplorasi.",
    "Penerima hadiah adalah kepala seksi yang tidak punya kewenangan langsung, namun memiliki akses terhadap dokumen evaluasi penawaran senilai Rp {amount}.",
    "Seorang pejabat daerah mendapatkan kendaraan bermotor senilai Rp {amount} dari developer perumahan yang proyeknya sedang dalam tinjauan tata ruang.",
    "Pengusaha menyerahkan cek senilai Rp {amount} kepada pejabat pada momen perayaan hari ulang tahun instansi pemerintah.",
    "Rekanan lama memberikan {gift} senilai Rp {amount} kepada pejabat yang baru dipromosikan ke jabatan yang berwenang menandatangani kontrak baru.",
]

# --- Bukan Milik Negara -----------------------------------------------------
BUKAN_MILIK_NEGARA_TEMPLATES = [
    # Family gifts
    "Orang tua memberikan uang tunai Rp {amount} kepada anak sebagai hadiah kelulusan studi doktoral.",
    "Suami memberikan {gift} senilai Rp {amount} kepada istrinya sebagai hadiah ulang tahun pernikahan.",
    "Kakak kandung mengirimkan transfer dana Rp {amount} kepada adiknya yang sedang sakit untuk biaya pengobatan.",
    "Orang tua mempelai perempuan memberikan {gift} senilai Rp {amount} kepada pasangan yang baru menikah sebagai modal rumah tangga.",
    "Paman memberikan {gift} senilai Rp {amount} kepada keponakannya yang baru diwisuda sebagai bentuk dukungan keluarga.",
    "Seorang ayah memberikan kendaraan senilai Rp {amount} kepada anaknya yang baru lulus seleksi CPNS sebagai hadiah kelulusan.",
    "Nenek memberikan perhiasan senilai Rp {amount} kepada cucunya pada acara akikah bayi.",
    "Anggota keluarga besar memberikan amplop berisi Rp {amount} pada pernikahan salah satu anggota keluarga.",
    "Kakak ipar memberikan {gift} senilai Rp {amount} kepada saudaranya sebagai hadiah atas kelahiran bayi pertama.",
    "Ibu kandung mentransfer Rp {amount} kepada anaknya yang bekerja sebagai ASN untuk biaya pengobatan cucunya.",

    # Close friends / social gifts
    "Teman dekat sejak SMA memberikan {gift} senilai Rp {amount} kepada sahabatnya yang baru pindah rumah sebagai kado housewarming.",
    "Kolega lama di kampus memberikan {gift} senilai Rp {amount} sebagai kenang-kenangan pada reuni alumni.",
    "Tetangga dekat memberikan bingkisan senilai Rp {amount} sebagai ucapan selamat atas promosi jabatan yang diterima.",
    "Sahabat memberikan {gift} senilai Rp {amount} kepada temannya yang baru sembuh dari sakit berat.",
    "Teman kerja menyumbang Rp {amount} secara kolektif untuk membantu biaya pernikahan rekan sejawat.",

    # Private / personal transactions
    "Karyawan swasta menerima bonus kinerja tahunan senilai Rp {amount} dari perusahaan tempat ia bekerja sesuai kontrak kerja.",
    "Direktur perusahaan menerima dividen senilai Rp {amount} dari saham yang dimilikinya di perusahaan milik sendiri.",
    "Pegawai BUMN menerima tunjangan hari raya senilai Rp {amount} yang telah dianggarkan dalam RKAP perusahaan.",
    "Seseorang menerima warisan berupa tanah senilai Rp {amount} dari mendiang orang tuanya berdasarkan surat wasiat.",
    "Karyawan menerima uang pensiun bulanan Rp {amount} setelah masa kerja berakhir sesuai peraturan perusahaan.",

    # Within policy / normal business
    "Perusahaan memberikan souvenir berlogo korporat senilai Rp {amount} kepada semua peserta pameran dagang sebagai bagian dari strategi promosi.",
    "Penyelenggara konferensi nasional memberikan goodie bag senilai Rp {amount} kepada seluruh peserta tanpa pandang jabatan.",
    "Pelanggan setia mendapatkan hadiah program loyalitas senilai Rp {amount} sesuai poin yang terkumpul dari transaksi.",
    "Bank memberikan bunga tabungan senilai Rp {amount} kepada nasabah atas saldo yang tersimpan sesuai perjanjian.",
    "Asuransi mencairkan klaim senilai Rp {amount} kepada tertanggung setelah verifikasi dokumen kecelakaan.",

    # Social/religious context
    "Peserta pengajian mendapatkan amplop rezeki senilai Rp {amount} dari penyelenggara dalam rangka peringatan Maulid Nabi.",
    "Jemaah haji menerima sumbangan Rp {amount} dari paguyuban kampung halaman sebagai bekal keberangkatan ke Tanah Suci.",
    "Warga menerima bantuan sosial tunai Rp {amount} melalui program resmi pemerintah pusat yang telah dianggarkan dalam APBN.",
    "Peserta program magang menerima uang saku Rp {amount} per bulan dari perusahaan yang menyelenggarakan program tersebut.",
    "Mahasiswa mendapatkan beasiswa prestasi Rp {amount} dari yayasan pendidikan swasta berdasarkan hasil seleksi akademik.",

    # Partner / vendor – no conflict of interest
    "Rekan bisnis sesama pengusaha swasta memberikan {gift} senilai Rp {amount} pada perayaan ulang tahun perusahaan mitra.",
    "Klien korporasi memberikan {gift} senilai Rp {amount} kepada konsultan swasta yang berhasil menyelesaikan proyek riset pasar.",
    "Perusahaan asuransi memberikan voucher makan senilai Rp {amount} kepada agen terbaik atas pencapaian target penjualan.",
    "Perusahaan teknologi memberikan lisensi perangkat lunak senilai Rp {amount} kepada universitas swasta sebagai bentuk CSR.",
    "Distributor memberikan diskon volume senilai Rp {amount} kepada pelanggan korporat yang memenuhi target pembelian kuartalan.",

    # Subtle / complex cases
    "Pegawai negeri menerima {gift} senilai Rp {amount} dari kakak kandung yang kebetulan berprofesi sebagai pengacara, bukan dari klien kakaknya.",
    "Seorang pejabat menerima {gift} dari teman lamanya senilai Rp {amount} yang diberikan pada acara perpisahan karena teman tersebut pindah ke luar negeri.",
    "Pemberian {gift} senilai Rp {amount} berasal dari sepupu yang tidak memiliki urusan atau kepentingan apapun dengan instansi tempat si pejabat bekerja.",
    "Kado ulang tahun senilai Rp {amount} yang diberikan oleh sesama pegawai negeri di unit yang sama tanpa keterkaitan dengan jabatan.",
    "Seorang pensiunan PNS menerima {gift} senilai Rp {amount} dari mantan rekan kerja sebagai tanda persahabatan di luar konteks jabatan.",
    "Hadiah senilai Rp {amount} diberikan oleh komunitas hobi yang tidak memiliki kaitan dengan pekerjaan atau kewenangan penerima.",
    "Kerabat jauh memberikan tanah warisan keluarga senilai Rp {amount} kepada pejabat berdasarkan garis keturunan tanpa konflik kepentingan.",
    "Teman masa kecil memberikan {gift} senilai Rp {amount} kepada temannya yang kebetulan menjadi PNS, tanpa urusan pekerjaan apapun.",
    "Asosiasi profesi memberikan penghargaan berupa plakat dan uang tunai Rp {amount} kepada anggota terbaik berdasarkan penilaian independen.",
    "Perusahaan asuransi swasta mencairkan manfaat polis jiwa senilai Rp {amount} kepada ahli waris atas kematian tertanggung.",
]

# Filler data for template placeholders
GIFTS = [
    "jam tangan mewah", "tas kulit branded", "perhiasan emas", "paket wisata",
    "voucher belanja", "smartphone terbaru", "laptop premium", "sepeda listrik",
    "wine impor", "hampers premium", "cek perjalanan", "kartu hadiah",
    "perangkat elektronik", "set peralatan golf", "kamera digital",
    "parfum mewah", "dompet kulit buaya", "koper branded",
    "jam tangan Swiss", "bingkisan lebaran mewah",
]

VENDORS = [
    "PT Maju Bersama", "CV Karya Mandiri", "PT Solusi Teknologi Nusantara",
    "PT Global Infrastruktur", "CV Bangun Sejahtera", "PT Anugerah Abadi",
    "PT Prima Konstruksi", "CV Bintang Utama", "PT Citra Logistik",
    "PT Dinamika Persada", "PT Bumi Perkasa", "CV Artha Mulia",
    "PT Mega Niaga Internasional", "PT Fortuna Konsultan", "PT Sigma Inovasi",
]

ITEMS = [
    "alat kesehatan", "perangkat TI", "kendaraan dinas", "seragam", "mebel kantor",
    "ATK", "jasa keamanan", "jasa kebersihan", "mesin cetak", "genset",
    "sistem CCTV", "software akuntansi", "layanan cloud", "bahan bakar",
    "lift gedung kantor",
]

PROJECTS = [
    "pembangunan gedung kantor baru", "renovasi jalan desa",
    "pengadaan sistem informasi daerah", "pembangunan rumah sakit daerah",
    "infrastruktur jaringan fiber optik", "revitalisasi pasar tradisional",
    "pembangunan bendungan irigasi", "pengadaan kendaraan operasional",
    "pembangunan dermaga nelayan", "modernisasi sistem perpajakan daerah",
]

CITIES = [
    "Singapura", "Tokyo", "Dubai", "Sydney", "London",
    "Bali", "Lombok", "Yogyakarta", "Surabaya", "Makassar",
    "Frankfurt", "Bangkok", "Seoul", "Amsterdam", "Paris",
]

AMOUNTS = [
    "250.000", "500.000", "750.000", "1.000.000", "1.500.000",
    "2.000.000", "2.500.000", "3.000.000", "5.000.000", "7.500.000",
    "10.000.000", "15.000.000", "20.000.000", "25.000.000", "30.000.000",
    "50.000.000", "75.000.000", "100.000.000", "150.000.000", "200.000.000",
    "300.000.000", "500.000.000", "750.000.000", "1.000.000.000",
]


def fill(template: str, rng: random.Random) -> str:
    """Fill template placeholders with random values."""
    result = template
    if "{gift}" in result:
        result = result.replace("{gift}", rng.choice(GIFTS), 1)
    if "{vendor}" in result:
        result = result.replace("{vendor}", rng.choice(VENDORS), 1)
    if "{item}" in result:
        result = result.replace("{item}", rng.choice(ITEMS), 1)
    if "{project}" in result:
        result = result.replace("{project}", rng.choice(PROJECTS), 1)
    if "{city}" in result:
        result = result.replace("{city}", rng.choice(CITIES), 1)
    if "{amount}" in result:
        result = result.replace("{amount}", rng.choice(AMOUNTS), 1)
    return result


def generate_samples(n_total: int, seed: int) -> list[dict]:
    """Generate n_total samples with balanced label distribution."""
    rng = random.Random(seed)

    n_each = n_total // 2  # 250 each for balanced distribution

    samples: list[dict] = []
    seen_texts: set[str] = set()

    def _generate_unique(templates: list[str], label: str, count: int) -> None:
        shuffled = templates.copy()
        rng.shuffle(shuffled)
        idx = 0
        retries = 0
        generated = 0
        while generated < count:
            template = shuffled[idx % len(shuffled)]
            text = fill(template, rng)
            idx += 1
            if text in seen_texts:
                retries += 1
                if retries > count * 10:
                    break
                continue
            retries = 0
            seen_texts.add(text)
            samples.append({"text": text, "label": label})
            generated += 1

    _generate_unique(MILIK_NEGARA_TEMPLATES, "Milik Negara", n_each)
    _generate_unique(BUKAN_MILIK_NEGARA_TEMPLATES, "Bukan Milik Negara", n_each)

    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic JSONL training data for Gratifikasi classifier."
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_training_data.jsonl",
        help="Output JSONL file path (default: data/synthetic_training_data.jsonl)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Total number of samples to generate (default: 500, must be even)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.samples % 2 != 0:
        raise ValueError("--samples must be an even number for balanced distribution.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(args.samples, args.seed)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    label_counts: dict[str, int] = {}
    for s in samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

    print(f"Generated {len(samples)} samples -> {output_path}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
