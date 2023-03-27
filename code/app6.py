"""
problématique de l'APP6 (S2)
Université de Sherbrooke
Hiver 2023

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import helpers as hp

def filtre_Passe_Bas():
    wc = 700*2* np.pi
    b, a = signal.butter(2, wc, 'low', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    print(f' Zéros:{z}, Pôles:{p}')

    # affiche zeros,poles
    hp.pzmap1(z, p, 'zeros et poles')
    #transfert = signal.TransferFunction()
    tf1 = signal.TransferFunction(b, a)
    w1,mag1, phlin1 = signal.bode(tf1)
    hp.bode1(w1,mag1, phlin1, 'Example 1')

    hp.pzmap1(z, p, 'bode final')
    mag, ph, w, fig, ax = hp.bodeplot(b, a, 'bode final')
    hp.grpdel1(w, -np.diff(ph) / np.diff(w), 'bode final')


def filtre_Passe_Haut():
    wc = 7000 * 2 * np.pi
    b, a = signal.butter(2, wc, 'high', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    print(f' Zéros:{z}, Pôles:{p}')

    # affiche zeros,poles
    hp.pzmap1(z, p, 'zeros et poles')
    # transfert = signal.TransferFunction()

def filtre_Passe_Haut_dePasseBande():
    wc = 2000 * np.pi
    b, a = signal.butter(2, wc, 'high', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    print(f' Zéros:{z}, Pôles:{p}')

    # affiche zeros,poles
    hp.pzmap1(z, p, 'zeros et poles')
    #transfert = signal.TransferFunction()

def filtre_Passe_Bas_dePasseBande():
    wc = 5000*2* np.pi
    b, a = signal.butter(2, wc, 'low', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    print(f' Zéros:{z}, Pôles:{p}')

    # affiche zeros,poles
    hp.pzmap1(z, p, 'zeros et poles')
    # transfert = signal.TransferFunction()

def signal_1():
    #passe haut de passe bande
    wc_haut = 2000 * np.pi
    b_haut, a_haut = signal.butter(2, wc_haut, 'high', analog=True)
    z_haut, p_haut, k_haut = signal.tf2zpk(b_haut, a_haut)
    #fonction de transfert passe haut
    tf_haut= signal.TransferFunction(b_haut, a_haut)
    w_haut, mag_haut, phlin_haut = signal.bode(tf_haut,2500*2*np.pi,2)
    print(mag_haut)
    #passe bas de passe bande
    wc_bas = 5000 * 2 * np.pi
    b_bas, a_bas = signal.butter(2, wc_bas, 'low', analog=True)
    z_bas, p_bas, k_bas = signal.tf2zpk(b_bas, a_bas)
    #fonction de transfert passe bas
    tf_bas= signal.TransferFunction(b_bas, a_bas)
    w_bas, mag_bas, phlin_bas = signal.bode(tf_bas,2500*2*np.pi,2)
    print(mag_bas)
    #valeur de gain
    g_total=(10**(mag_haut/20)*10**(mag_bas/20))
    print(0.25*g_total)
    #valeur de phase
    ph_total=(phlin_haut*2*np.pi/360)+(phlin_bas*2*np.pi/360)
    print(ph_total)
    #signal
    f_sin=2500 #Hz
    t, step = np.linspace(0, 0.0025, 5000, retstep=True)
    u=0.25*np.sin(2*np.pi*f_sin*t)
    #gain
    g_haut=1
    g_bas=1
    #génère série
    zs, ps, ks = hp.seriestf(z_haut, p_haut, k_haut * g_haut, z_bas, p_bas, k_bas * g_bas)
    bs, a_s = signal.zpk2tf(zs, ps, ks)
    print(f'Égaliseur klow={g_bas}, khigh={g_haut}: Zéros : {zs}, Pôles: {ps}, Gain: {ks}')
    mags, phs, ws, fig, ax = hp.bodeplot(bs, a_s, f'klow={g_bas}, khigh={g_haut}')
    hp.grpdel1(ws, -np.diff(phs) / np.diff(ws), f'klow={g_bas}, khigh={g_haut}')
    # simule
    touts, youts, xouts = signal.lsim((bs, a_s), u, t)
    tout_haut, yout_haut, xout_haut = signal.lsim((z_haut, p_haut, k_haut * g_haut), u, t)
    tout_bas, yout_bas, xout_bas = signal.lsim((z_bas, p_bas, k_bas * g_bas), u, t)
    yout = [yout_haut, yout_bas, youts]
    hp.timepltmulti2(t, u, touts, yout, f'Égaliseur klow={g_bas}, khigh={g_haut}', ['H1', 'H2', 'HÉgaliseur'])
    return g_total,ph_total;

def lieu_de_bode_circuit_corrigé():
    #passe bas
    wc_bas = 700 * 2 * np.pi
    b_bas, a_bas = signal.butter(2, wc_bas, 'low', analog=True)
    (z_bas, p_bas, k_bas) = signal.tf2zpk(b_bas, a_bas)
    mag_bas, ph_bas, w_bas, fig, ax = hp.bodeplot(b_bas, a_bas, 'bode final')
    #passe haut
    wc_haut = 7000 * 2 * np.pi
    b_haut, a_haut = signal.butter(2, wc_haut, 'high', analog=True)
    (z_haut, p_haut, k_haut) = signal.tf2zpk(b_haut, a_haut)

    #passe haut de bande
    wc_haut_bande = 2000 * np.pi
    b_haut_bande, a_haut_bande = signal.butter(2, wc_haut_bande, 'high', analog=True)
    (z_haut_bande, p_haut_bande, k_haut_bande) = signal.tf2zpk(b_haut_bande, a_haut_bande)

    #passe bas de bande
    wc_bas_bande = 5000*2* np.pi
    b_bas_bande, a_bas_bande = signal.butter(2, wc_bas_bande, 'low', analog=True)
    (z_bas_bande, p_bas_bande, k_bas_bande) = signal.tf2zpk(b_bas_bande, a_bas_bande)

    #parallele bas et haut
    zp1, pp1, kp1 = hp.paratf(z_bas, p_bas, k_bas, z_haut, p_haut, k_haut)
    bp1, ap1 = signal.zpk2tf(zp1, pp1, kp1)

    #serie bande
    zs, ps, ks = hp.seriestf(z_bas_bande, p_bas_bande, k_bas_bande, z_haut_bande, p_haut_bande, k_haut_bande)
    bs, a_s = signal.zpk2tf(zs, ps, ks)
    ks=ks*100
    #hp.pzmap1(zs, ps, 'bode bande')
    mags, phs, ws, fig, ax = hp.bodeplot(bs, a_s, 'bode bande')
    #hp.grpdel1(ws, -np.diff(phs) / np.diff(ws), 'bode bande')

    #parallele totale
    zp2, pp2, kp2 = hp.paratf(zp1, pp1, kp1, zs, ps, ks)
    bp2, ap2 = signal.zpk2tf(zp2, pp2, kp2)
    print(f'H1+H2 Zéros : {zp2}, Pôles: {pp2}, Gain: {kp2}')
    hp.pzmap1(zp2, pp2, 'bode final')
    magp2, php2, wp2, fig, ax = hp.bodeplot(bp2, ap2, 'bode final')
    hp.grpdel1(wp2, -np.diff(php2) / np.diff(wp2), 'bode final')
def main():
    #filtre_Passe_Haut_dePasseBande()
    filtre_Passe_Bas()
    #filtre_Passe_Haut()
    #filtre_Passe_Bas_dePasseBande()
    #signal_1()
    #lieu_de_bode_circuit_corrigé()
    plt.show()

if __name__ == '__main__':
    main()