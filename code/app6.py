"""
problématique de l'APP6 (S2)
Université de Sherbrooke
Hiver 2023

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import helpers as hp

def paratf_ba(b1,a1,b2,a2):
    bleft = np.convolve(b1, a2) # calcule les 2 termes du numérateur
    bright = np.convolve(b2, a1)
    b = np.polyadd(bleft, bright)
    a = np.convolve(a1, a2)
    z, p, k = signal.tf2zpk(b, a)
    z, p, k = hp.simplifytf(z, p, k)
    return z, p, k

def filtre_Passe_Bas():
    wc = 700*2* np.pi
    b, a = signal.butter(2, wc, 'low', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    return b,a,z,p,k

def filtre_Passe_Haut():
    wc = 7000 * 2 * np.pi
    b, a = signal.butter(2, wc, 'high', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    return b,a,z,p,k

def filtre_Passe_Haut_dePasseBande():
    wc = 2000 * np.pi
    b, a = signal.butter(2, wc, 'high', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    print(f' Zéros:{z}, Pôles:{p}')

    # affiche zeros,poles
    hp.pzmap1(z,p,'passe haut de bande')

    # affiche lieu de bode
    tf = signal.TransferFunction(b, a)
    w, mag, phlin = signal.bode(tf)
    hp.bode1(w, mag, phlin, 'passe haut de bande ')
    return b,a,z,p,k

def filtre_Passe_Bas_dePasseBande():
    wc = 5000*2* np.pi
    b, a = signal.butter(2, wc, 'low', analog=True)
    print(f'Butterworth Numérateur {b}, Dénominateur {a}')
    z, p, k = signal.tf2zpk(b, a)
    return b,a,z,p,k

def signal_1():
    #passe haut de passe bande
    b_haut,a_haut,p_haut,z_haut,k_haut=filtre_Passe_Haut_dePasseBande()

    #fonction de transfert passe haut
    tf_haut= signal.TransferFunction(b_haut, a_haut)
    w_haut, mag_haut, phlin_haut = signal.bode(tf_haut,2500*2*np.pi,2)

    #passe bas de passe bande
    b_bas,a_bas,p_bas,z_bas,k_bas=filtre_Passe_Bas_dePasseBande()

    #fonction de transfert passe bas
    tf_bas= signal.TransferFunction(b_bas, a_bas)
    w_bas, mag_bas, phlin_bas = signal.bode(tf_bas,2500*2*np.pi,2)

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
    b_bas,a_bas,z_bas,p_bas,k_bas = filtre_Passe_Bas()

    #passe haut
    b_haut,a_haut,z_haut,p_haut,k_haut = filtre_Passe_Haut()

    #passe haut de bande
    b_haut_bande,a_haut_bande,z_haut_bande,p_haut_bande,k_haut_bande = filtre_Passe_Haut_dePasseBande()

    #passe bas de bande
    b_bas_bande, a_bas_bande, z_bas_bande, p_bas_bande, k_bas_bande = filtre_Passe_Bas_dePasseBande()

    #parallele bas et haut
    zp1, pp1, kp1 = paratf_ba(-1*b_bas, a_bas, -1*b_haut, a_haut)
    bp1, ap1 = signal.zpk2tf(zp1, pp1, kp1)

    deltaDB = 10
    topGain = 2
    lowGain = 0.2
    topMag = 10
    lowMag = -10
    finalGain = 1
    for itter in range(0, 20):
        gain = (topGain+lowGain)/2
        zs, ps, ks = hp.seriestf(z_bas_bande, p_bas_bande, gain * k_bas_bande, z_haut_bande, p_haut_bande, k_haut_bande)

        # parallele totale
        zp2, pp2, kp2 = hp.paratf(zp1, pp1, kp1, zs, ps, ks)
        bp2, ap2 = signal.zpk2tf(zp2, pp2, kp2)
        # magp2, php2, wp2, fig, ax = hp.bodeplot(bp2, ap2, str(gain))
        # magnitude (dB)
        w, h = signal.freqs(bp2, ap2,
                            5000)  # calcul la réponse en fréquence du filtre (H(jw)), fréquence donnée en rad/sec
        mag = 20 * np.log10(np.abs(h))
        #newDeltaDB = max(mag) - min(mag)

        if( (max(mag) + min(mag)) > 0):
            if (max(mag) < topMag):
                finalGain = gain
                topGain = gain
                topMag = max(mag)
        else:
            if (min(mag) > lowMag):
                finalGain = gain
                lowGain = gain
                lowMag = min(mag)


        # newDeltaDB = max(mag) - min(mag)
        #print("newDeltaDB: ", newDeltaDB, " with gain: ", gain)
        print("topGain: ", topGain, " lowGain: ", lowGain)
        # newTop = max(magp2)
        # newLow = min(magp2)

    #serie bande
    k2 = finalGain
    # k2 = 0.8
    zs, ps, ks = hp.seriestf(z_bas_bande, p_bas_bande, k2*k_bas_bande, z_haut_bande, p_haut_bande, k_haut_bande)
    bs, a_s = signal.zpk2tf(zs, ps, ks)

    #parallele totale
    zp2, pp2, kp2 = hp.paratf(zp1, pp1, kp1, zs, ps, ks)
    bp2, ap2 = signal.zpk2tf(zp2, pp2, kp2)
    magp2, php2, wp2, fig, ax = hp.bodeplot(bp2, ap2, 'Lieu de Bode - ')
    hp.grpdel1(wp2, -np.diff(php2) / np.diff(wp2), '- Complete circuit')



def main():
    #filtre_Passe_Haut_dePasseBande()
    #filtre_Passe_Bas()
    #filtre_Passe_Haut()
    #filtre_Passe_Bas_dePasseBande()
    #signal_1()
    lieu_de_bode_circuit_corrigé()
    plt.show()

if __name__ == '__main__':
    main()