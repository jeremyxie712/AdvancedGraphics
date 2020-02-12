import matplotlib.pyplot as plt
import math
import numpy as np


def report_brewster(eta_i,eta_t):
    return math.atan(eta_t/eta_i)

def report_critical(eta_i,eta_t):
    return math.asin(eta_t/eta_i)

def schlick_approximation(theta_i,r_0):
    return r_0 + (1.-r_0)*((1. - math.cos(theta_i))**5.)

def index_of_refraction(n,k):
    theta_deg = 0

    fresnel = []
    while theta_deg <= 90:
        theta = math.radians(theta_deg)

        a = math.sqrt((math.sqrt((n**2-k**2-(math.sin(theta))**2)**2 +
                                 ((4 * n**2) * k**2)) + (n**2 - k**2 -
                                            (math.sin(theta))**2))/2)

        b = math.sqrt((math.sqrt((n**2-k**2-(math.sin(theta))**2)**2 +
                                 ((4 * n**2) * k**2)) - (n**2 - k**2 -
                                            (math.sin(theta))**2))/2)

        Fs = (a**2+b**2-(2 * a * math.cos(theta))+
              (math.cos(theta))**2)/(a**2+b**2 +
                            (2 * a * math.cos(theta))+(math.cos(theta))**2)

        Fp = Fs * ((a**2+b**2 -
                    (2 * a * math.sin(theta) * math.tan(theta)) +
                    (math.sin(theta))**2*(math.tan(theta))**2)/(a**2+b**2 +
                    (2 * a * math.sin(theta) * math.tan(theta)) +
                    (math.sin(theta))**2*(math.tan(theta))**2))

        R = (Fs + Fp)/2

        fresnel.append((R, Fs, Fp))

        theta_deg += 1
    return fresnel


def fresnel_function(eta_i,eta_t,theta_i):
    theta_t = math.asin((eta_i/eta_t)*math.sin(theta_i))
    
    p_polarised_reflectance = ((eta_t*math.cos(theta_i)-eta_i*math.cos(theta_t))/(eta_t*math.cos(theta_i)+eta_i*math.cos(theta_t))) ** 2
    s_polarised_reflectance = ((eta_i*math.cos(theta_i)-eta_t*math.cos(theta_t))/(eta_i*math.cos(theta_i)+eta_t*math.cos(theta_t))) ** 2

    Fr = (0.5) * (p_polarised_reflectance + s_polarised_reflectance)
    Tr = 1 - Fr
    return p_polarised_reflectance,s_polarised_reflectance,Fr,Tr

def main(eta_i,eta_t):
    plt.figure()
    if eta_i < eta_t:
        bound = 90
        brewster = math.degrees(report_brewster(eta_i,eta_t))
        deg_points  = np.linspace(0,bound,100)
        plt.vlines(brewster, 0., 1., colors='y', linestyles="dashed",label='Brewster Angle')
    else:
        critical = math.degrees(report_critical(eta_i,eta_t))
        deg_points  = np.linspace(0,critical,100)
        plt.vlines(critical, 0., 1., colors='m', linestyles="dashed",label='Critical Angle')
    p_polarised = np.zeros(100)
    s_polarised = np.zeros(100)
    reflectance = np.zeros(100)
    transmittance=np.zeros(100)
    sch_approx  = np.zeros(100)

    r_0 = fresnel_function(eta_i,eta_t,0)[2]
    i = 0
    while i < 100:
        theta = math.radians(deg_points[i])
        p_polarised[i],s_polarised[i],reflectance[i],transmittance[i] = fresnel_function(eta_i,eta_t,theta)
        sch_approx[i] = schlick_approximation(np.arcsin(max(1, (eta_i/eta_t)) * np.sin(theta)), r_0)
        i += 1

    plt.plot(deg_points,p_polarised,'-k',linewidth=1,label='p-polarised')
    plt.plot(deg_points,s_polarised,'--b',linewidth=1,label='s-polarised')
    plt.plot(deg_points,reflectance,'-r',linewidth=1,label='Unpolarised')
    plt.plot(deg_points,transmittance,'-g',linewidth=1,label='transmittance')
    plt.plot(deg_points,sch_approx,'-c',linewidth=1,label='Schlick Approximation')
    plt.xlim(0,90)
    plt.ylim(0,1)
    plt.xlabel('Degrees')
    plt.ylabel('Reflectance')
    plt.title('$\eta_i$ = 1.00 $\eta_t$ = 1.45')
    plt.legend()
    plt.show()


main(1.45,1.)