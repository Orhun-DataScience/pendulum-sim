"""
Basit Sarkaç (Simple Pendulum) Simülasyonu
==========================================
Parametreler: L = 1.5 m | m = 0.5 kg | theta0 = 15°
Yöntem: RK4 (Runge-Kutta 4. derece) nümerik integrasyon

Pencereler:
  Sol üst  → Sarkaç animasyonu (gerçek zamanlı)
  Sağ üst  → Açı vs Zaman (θ)
  Sol alt  → Hız vs Zaman (ω)
  Sağ alt  → Enerji vs Zaman (KE, PE, Toplam)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ── Fiziksel Parametreler ──────────────────────────────────────────────────
L   = 1.5          # ip uzunluğu (m)
m   = 0.5          # kütle (kg)
g   = 9.81         # yerçekimi ivmesi (m/s²)
th0 = np.radians(15)  # başlangıç açısı (rad)
w0  = 0.0          # başlangıç açısal hızı (rad/s)
b   = 0.0          # sönümleme katsayısı (0 = sönümsüz)

T_sim  = 10.0      # simülasyon süresi (s)
dt     = 0.01      # zaman adımı (s)
t_arr  = np.arange(0, T_sim, dt)
N      = len(t_arr)

# ── Teorik Değerler ────────────────────────────────────────────────────────
omega_th = np.sqrt(g / L)          # açısal frekans (rad/s)
T_th     = 2 * np.pi / omega_th    # periyot (s)
h_max    = L * (1 - np.cos(th0))   # maksimum yükseklik farkı (m)
v_max    = np.sqrt(2 * g * h_max)  # maksimum hız (m/s)

# ── RK4 Entegratörü ────────────────────────────────────────────────────────
def dydx(t, y):
    theta, omega = y
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - b * omega
    return np.array([dtheta, domega])

def rk4_step(t, y, h):
    k1 = dydx(t,       y)
    k2 = dydx(t + h/2, y + h/2 * k1)
    k3 = dydx(t + h/2, y + h/2 * k2)
    k4 = dydx(t + h,   y + h   * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ── Simülasyonu Çalıştır ───────────────────────────────────────────────────
theta_arr = np.zeros(N)
omega_arr = np.zeros(N)
theta_arr[0] = th0
omega_arr[0] = w0

for i in range(N - 1):
    state = np.array([theta_arr[i], omega_arr[i]])
    new   = rk4_step(t_arr[i], state, dt)
    theta_arr[i+1] = new[0]
    omega_arr[i+1] = new[1]

# ── Enerji Hesabı ──────────────────────────────────────────────────────────
v_arr  = L * omega_arr
KE_arr = 0.5 * m * v_arr**2
PE_arr = m * g * L * (1 - np.cos(theta_arr))
E_arr  = KE_arr + PE_arr

# Top koordinatları
x_arr = L * np.sin(theta_arr)
y_arr = -L * np.cos(theta_arr)

# ── Küçük Açı Analitik Çözüm (karşılaştırma için) ─────────────────────────
theta_analytic = th0 * np.cos(omega_th * t_arr)

# ══════════════════════════════════════════════════════════════════════════
# Şekil Düzeni
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 9), facecolor='#0f0f1a')
fig.suptitle('Basit Sarkaç Simülasyonu  |  L=1.5m  m=0.5kg  θ₀=15°',
             color='white', fontsize=14, fontweight='bold', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3,
              left=0.07, right=0.96, top=0.93, bottom=0.07)

ax_pend = fig.add_subplot(gs[0, 0])  # sarkaç animasyonu
ax_th   = fig.add_subplot(gs[0, 1])  # açı
ax_om   = fig.add_subplot(gs[1, 0])  # açısal hız
ax_en   = fig.add_subplot(gs[1, 1])  # enerji

DARK_BG = '#0f0f1a'
PANEL   = '#1a1a2e'
for ax in (ax_pend, ax_th, ax_om, ax_en):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#aaaacc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# ── Sol Üst: Sarkaç Panosu ─────────────────────────────────────────────────
pad = 0.25
ax_pend.set_xlim(-L - pad, L + pad)
ax_pend.set_ylim(-L - pad, 0.4)
ax_pend.set_aspect('equal')
ax_pend.set_title('Sarkaç Hareketi', color='white', fontsize=10, pad=6)
ax_pend.axhline(0, color='#444466', lw=1, ls='--')

# Sabit nokta
ax_pend.plot(0, 0, 's', color='#aaaacc', markersize=8, zorder=5)

# İz (gölge yolu)
trail_line, = ax_pend.plot([], [], '-', color='#3355ff', alpha=0.3, lw=1)

# İp
rope_line, = ax_pend.plot([], [], '-', color='#ccccee', lw=2)

# Top
ball_circ = plt.Circle((0, -L), 0.07, color='#ff5555', zorder=6)
ax_pend.add_patch(ball_circ)

# Bilgi kutusu
info_text = ax_pend.text(
    -L - pad + 0.05, 0.32,
    '', color='#aaffaa', fontsize=8,
    fontfamily='monospace', va='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a18', alpha=0.7)
)

# Yay hareketi göstergesi (açı yayı)
angle_arc = patches.Arc((0, 0), 0.6, 0.6, angle=0,
                         theta1=-90, theta2=-90,
                         color='#ffff66', lw=1.5)
ax_pend.add_patch(angle_arc)

# ── Sağ Üst: Açı Grafiği ──────────────────────────────────────────────────
ax_th.set_title('Açı  θ(t)', color='white', fontsize=10, pad=6)
ax_th.set_xlabel('Zaman (s)', color='#aaaacc', fontsize=8)
ax_th.set_ylabel('θ (derece)', color='#aaaacc', fontsize=8)
ax_th.set_xlim(0, T_sim)
ax_th.set_ylim(-20, 20)
ax_th.axhline(0, color='#333355', lw=0.8)
ax_th.plot(t_arr, np.degrees(theta_analytic),
           '--', color='#ffaa00', lw=1, alpha=0.5, label='Küçük açı (analitik)')
line_th, = ax_th.plot([], [], '-', color='#55aaff', lw=1.8, label='RK4 (tam)')
ax_th.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333355',
             labelcolor='white', loc='upper right')

# Teorik periyot çizgileri
for k in range(1, int(T_sim / T_th) + 1):
    ax_th.axvline(k * T_th, color='#ff5555', lw=0.6, alpha=0.4, ls=':')

# ── Sol Alt: Açısal Hız ────────────────────────────────────────────────────
ax_om.set_title('Açısal Hız  ω(t)', color='white', fontsize=10, pad=6)
ax_om.set_xlabel('Zaman (s)', color='#aaaacc', fontsize=8)
ax_om.set_ylabel('ω (rad/s)', color='#aaaacc', fontsize=8)
ax_om.set_xlim(0, T_sim)
ax_om.set_ylim(-1.2, 1.2)
ax_om.axhline(0, color='#333355', lw=0.8)
line_om, = ax_om.plot([], [], '-', color='#ff8844', lw=1.8)

# ── Sağ Alt: Enerji ────────────────────────────────────────────────────────
ax_en.set_title('Enerji (J)', color='white', fontsize=10, pad=6)
ax_en.set_xlabel('Zaman (s)', color='#aaaacc', fontsize=8)
ax_en.set_ylabel('Enerji (J)', color='#aaaacc', fontsize=8)
ax_en.set_xlim(0, T_sim)
E0 = E_arr[0]
ax_en.set_ylim(-0.01, E0 * 1.4)
line_ke, = ax_en.plot([], [], '-', color='#55ff88', lw=1.5, label='Kinetik (KE)')
line_pe, = ax_en.plot([], [], '-', color='#ff5577', lw=1.5, label='Potansiyel (PE)')
line_et, = ax_en.plot([], [], '-', color='#ffffff', lw=1.2, ls='--', label='Toplam')
ax_en.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333355',
             labelcolor='white', loc='upper right')

# ── Teorik Bilgi Paneli ────────────────────────────────────────────────────
theory_str = (
    f"T  = {T_th:.3f} s\n"
    f"ω  = {omega_th:.3f} rad/s\n"
    f"v_max = {v_max:.3f} m/s\n"
    f"E₀ = {E0:.4f} J"
)
fig.text(0.5, 0.01, f"Teorik  →  {theory_str.replace(chr(10), '   ')}",
         color='#aaffaa', fontsize=8, ha='center', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a18', alpha=0.6))

# ── İz Tamponu ────────────────────────────────────────────────────────────
TRAIL_LEN = 60
trail_x = []
trail_y = []

# ══════════════════════════════════════════════════════════════════════════
# Animasyon Güncelleme Fonksiyonu
# ══════════════════════════════════════════════════════════════════════════
def update(frame):
    i = frame

    # Sarkaç konumu
    bx, by = x_arr[i], y_arr[i]

    # İz
    trail_x.append(bx)
    trail_y.append(by)
    if len(trail_x) > TRAIL_LEN:
        trail_x.pop(0)
        trail_y.pop(0)
    trail_line.set_data(trail_x, trail_y)

    # İp ve top
    rope_line.set_data([0, bx], [0, by])
    ball_circ.set_center((bx, by))

    # Açı yayı
    deg = -np.degrees(theta_arr[i])
    angle_arc.theta1 = min(-90, -90 + deg) if deg < 0 else -90
    angle_arc.theta2 = max(-90, -90 + deg) if deg > 0 else -90

    # Bilgi kutusu
    t_now = t_arr[i]
    info_text.set_text(
        f"t    = {t_now:.2f} s\n"
        f"θ    = {np.degrees(theta_arr[i]):+.2f}°\n"
        f"ω    = {omega_arr[i]:+.3f} rad/s\n"
        f"v    = {abs(v_arr[i]):.3f} m/s\n"
        f"KE   = {KE_arr[i]:.4f} J\n"
        f"PE   = {PE_arr[i]:.4f} J"
    )

    # Açı grafiği
    line_th.set_data(t_arr[:i+1], np.degrees(theta_arr[:i+1]))

    # Açısal hız grafiği
    line_om.set_data(t_arr[:i+1], omega_arr[:i+1])

    # Enerji grafikleri
    line_ke.set_data(t_arr[:i+1], KE_arr[:i+1])
    line_pe.set_data(t_arr[:i+1], PE_arr[:i+1])
    line_et.set_data(t_arr[:i+1], E_arr[:i+1])

    return (trail_line, rope_line, ball_circ, info_text,
            line_th, line_om, line_ke, line_pe, line_et)

# ── Animasyonu Başlat ──────────────────────────────────────────────────────
INTERVAL = 20   # ms (≈50 fps)
STEP     = max(1, int((INTERVAL / 1000) / dt))  # kaç veri noktası atlanacak

anim = FuncAnimation(
    fig, update,
    frames=range(0, N, STEP),
    interval=INTERVAL,
    blit=True,
    repeat=True
)

plt.show()
