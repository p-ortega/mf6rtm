# Example 3 from PHT3D v1.0 manual

from pylab import *

pht3d = loadtxt('pht3d.sel', skiprows=1)
C_4 = loadtxt('PHT3D001.ACN')
Ca = loadtxt('PHT3D002.ACN')
Cl = loadtxt('PHT3D003.ACN')
Mg = loadtxt('PHT3D004.ACN')
Na = loadtxt('PHT3D005.ACN')
K = loadtxt('PHT3D006.ACN')
Fe_2 = pht3d[:, 0]
Fe_3 = pht3d[:, 1]
Mn_2 = loadtxt('PHT3D008.ACN')
Al = loadtxt('PHT3D009.ACN')
Si = loadtxt('PHT3D010.ACN')
S_6 = loadtxt('PHT3D011.ACN')
pH = loadtxt('PHT3D012.ACN')
pe = loadtxt('PHT3D013.ACN')

cc = loadtxt('PHT3D014.ACN')
sid = loadtxt('PHT3D015.ACN')
gyp = loadtxt('PHT3D016.ACN')
sio2 = loadtxt('PHT3D017.ACN')
gib =  loadtxt('PHT3D018.ACN')
feoh =  loadtxt('PHT3D019.ACN')

porosity = 0.35
cc = cc / porosity
sid = sid / porosity
gyp = gyp / porosity
sio2 = sio2 / porosity
gib = gib / porosity
feoh = feoh / porosity

Al_ms_1 = loadtxt('verif/ms_al_6.dat')
Al_ms_2 = loadtxt('verif/ms_al_12.dat')
Al_ms_3 = loadtxt('verif/ms_al_24.dat')

C_4_ms_1 = loadtxt('verif/ms_co3_6.dat')
C_4_ms_2 = loadtxt('verif/ms_co3_12.dat')
C_4_ms_3 = loadtxt('verif/ms_co3_24.dat')

Ca_ms_1 = loadtxt('verif/ms_ca_6.dat')
Ca_ms_2 = loadtxt('verif/ms_ca_12.dat')
Ca_ms_3 = loadtxt('verif/ms_ca_24.dat')

Fe_2_ms_1 = loadtxt('verif/ms_fe2_6.dat')
Fe_2_ms_2 = loadtxt('verif/ms_fe2_12.dat')
Fe_2_ms_3 = loadtxt('verif/ms_fe2_24.dat')

Fe_3_ms_1 = loadtxt('verif/ms_fe3_6.dat')
Fe_3_ms_2 = loadtxt('verif/ms_fe3_12.dat')
Fe_3_ms_3 = loadtxt('verif/ms_fe3_24.dat')

S_6_ms_1 = loadtxt('verif/ms_s6_6.dat')
S_6_ms_2 = loadtxt('verif/ms_s6_12.dat')
S_6_ms_3 = loadtxt('verif/ms_s6_24.dat')

pH_ms_1 = loadtxt('verif/ms_ph_6.dat')
pH_ms_2 = loadtxt('verif/ms_ph_12.dat')
pH_ms_3 = loadtxt('verif/ms_ph_24.dat')

pe_ms_1 = loadtxt('verif/ms_pe_6.dat')
pe_ms_2 = loadtxt('verif/ms_pe_12.dat')
pe_ms_3 = loadtxt('verif/ms_pe_24.dat')

cc_ms_1 = loadtxt('verif/ms_calc_6.dat')
cc_ms_2 = loadtxt('verif/ms_calc_12.dat')
cc_ms_3 = loadtxt('verif/ms_calc_24.dat')

gib_ms_1 = loadtxt('verif/ms_gibb_6.dat')
gib_ms_2 = loadtxt('verif/ms_gibb_12.dat')
gib_ms_3 = loadtxt('verif/ms_gibb_24.dat')

sid_ms_1 = loadtxt('verif/ms_sid_6.dat')
sid_ms_2 = loadtxt('verif/ms_sid_12.dat')
sid_ms_3 = loadtxt('verif/ms_sid_24.dat')

gyp_ms_1 = loadtxt('verif/ms_gyp_6.dat')
gyp_ms_2 = loadtxt('verif/ms_gyp_12.dat')
gyp_ms_3 = loadtxt('verif/ms_gyp_24.dat')

#phc = load('phreeqc/ex03.sel')
#C_4_phc = phc[:, 10]
#Ca_phc = phc[:, 6]
#Cl_phc = phc[:, 10]
#Mg_phc = phc[:, 10]
#Na_phc = phc[:, 10]
#K_phc = phc[:, 10]
#Fe_2_phc = phc[:, 7]
#Fe_3_phc = phc[:, 11]
#Mn_2_phc = phc[:, 10]
#Al_phc = phc[:, 8]
#Si_phc = phc[:, 10]
#S_6_phc = phc[:, 12]
#pH_phc = phc[:, 9]
#pe_phc = phc[:, 13]

#cc_phc = phc[:, 2]
#sid_phc = phc[:, 3]
#gyp_phc = phc[:, 5]
#sio2_phc = phc[:, 4]
#gib_phc = phc[:, 4]
#feoh_phc = phc[:, 4]

s1 = 0
e1 = 80
s2 = 80
e2 = 160
s3 = 160
e3 = 240

s1_phc = 80
e1_phc = 160
s2_phc = 160
e2_phc = 240
s3_phc = 320
e3_phc = 400

x = arange(0.0025, 0.40, .005)
x_ms = arange(0, 0.405, 0.005)

#close()
#ioff()
figure()

subplot(4,3,1)
plot(x, cc[s1:e1],'r-', x, cc[s2:e2],'g-', x, cc[s3:e3],'b-')
#plot(x, cc_phc[s1_phc:e1_phc],'g-', x, cc_phc[s2_phc:e2_phc],'g-', x, cc_phc[s3_phc:e3_phc],'g-')
plot(x_ms,cc_ms_1[:,1],'r.', x_ms, cc_ms_2[:,1],'g.', x_ms, cc_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.01, 0.02])
title('Calcite')

subplot(4,3,2)
plot(x, Ca[s1:e1],'r-', x, Ca[s2:e2],'g-', x, Ca[s3:e3],'b-')
#plot(x, Ca_phc[s1_phc:e1_phc],'g-', x, Ca_phc[s2_phc:e2_phc],'g-', x, Ca_phc[s3_phc:e3_phc],'g-')
plot(x_ms,Ca_ms_1[:,1],'r.', x_ms, Ca_ms_2[:,1],'g.', x_ms, Ca_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.009, 0.018])
title('Ca')

subplot(4,3,3)
plot(x, C_4[s1:e1],'r-', x, C_4[s2:e2],'g-', x, C_4[s3:e3],'b-')
#plot(x, C_4_phc[s1_phc:e1_phc],'g-', x, C_4_phc[s2_phc:e2_phc],'g-', x, C_4_phc[s3_phc:e3_phc],'g-')
plot(x_ms,C_4_ms_1[:,1],'r.', x_ms, C_4_ms_2[:,1],'g.', x_ms, C_4_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.005, 0.01])
title('C(4)')

subplot(4,3,4)
plot(x, sid[s1:e1],'r-', x, sid[s2:e2],'g-', x, sid[s3:e3],'b-')
#plot(x, sid_phc[s1_phc:e1_phc],'g-', x, sid_phc[s2_phc:e2_phc],'g-', x, sid_phc[s3_phc:e3_phc],'g-')
plot(x_ms,sid_ms_1[:,1],'r.', x_ms, sid_ms_2[:,1],'g.', x_ms, sid_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.0125, 0.025])
title('Siderite')

subplot(4,3,5)
semilogy(x, Fe_2[s1:e1],'r-', x, Fe_2[s2:e2],'g-', x, Fe_2[s3:e3],'b-')
#semilogy(x, Fe_2_phc[s1_phc:e1_phc],'g-', x, Fe_2_phc[s2_phc:e2_phc],'g-', x, Fe_2_phc[s3_phc:e3_phc],'g-')
semilogy(x_ms,Fe_2_ms_1[:,1],'r.', x_ms, Fe_2_ms_2[:,1],'g.', x_ms, Fe_2_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
title('Fe(2)')

subplot(4,3,6)
semilogy(x, Fe_3[s1:e1],'r-', x, Fe_3[s2:e2],'g-', x, Fe_3[s3:e3],'b-')
#semilogy(x, Fe_3_phc[s1_phc:e1_phc],'g-', x, Fe_3_phc[s2_phc:e2_phc],'g-', x, Fe_3_phc[s3_phc:e3_phc],'g-')
semilogy(x_ms,Fe_3_ms_1[:,1],'r.', x_ms, Fe_3_ms_2[:,1],'g.', x_ms, Fe_3_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
title('Fe(3)')

subplot(4,3,7)
plot(x, gib[s1:e1],'r-', x, gib[s2:e2],'g-', x, gib[s3:e3],'b-')
#plot(x, gib_phc[s1_phc:e1_phc],'g-', x, gib_phc[s2_phc:e2_phc],'g-', x, gib_phc[s3_phc:e3_phc],'g-')
plot(x_ms,gib_ms_1[:,1],'r.', x_ms, gib_ms_2[:,1],'g.', x_ms, gib_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.01, 0.02])
title('Gibbsite')

subplot(4,3,8)
semilogy(x, Al[s1:e1],'r-', x, Al[s2:e2],'g-', x, Al[s3:e3],'b-')
#semilogy(x, Al_phc[s1_phc:e1_phc],'g-', x, Al_phc[s2_phc:e2_phc],'g-', x, Al_phc[s3_phc:e3_phc],'g-')
semilogy(x_ms,Al_ms_1[:,1],'r.', x_ms, Al_ms_2[:,1],'g.', x_ms, Al_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
title('Al')

subplot(4,3,9)
plot(x, S_6[s1:e1],'r-', x, S_6[s2:e2],'g-', x, S_6[s3:e3],'b-')
#plot(x, S_6_phc[s1_phc:e1_phc],'g-', x, S_6_phc[s2_phc:e2_phc],'g-', x, S_6_phc[s3_phc:e3_phc],'g-')
plot(x_ms,S_6_ms_1[:,1],'r.', x_ms, S_6_ms_2[:,1],'g.', x_ms, S_6_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.03, 0.06])
title('S(6)')

subplot(4,3,10)
plot(x, gyp[s1:e1],'r-', x, gyp[s2:e2],'g-', x, gyp[s3:e3],'b-')
#plot(x, gyp_phc[s1_phc:e1_phc],'g-', x, gyp_phc[s2_phc:e2_phc],'g-', x, gyp_phc[s3_phc:e3_phc],'g-')
plot(x_ms,gyp_ms_1[:,1],'r.', x_ms, gyp_ms_2[:,1],'g.', x_ms, gyp_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(mol/l)')
yticks([0.0, 0.009, 0.018])
title('Gypsum')

subplot(4,3,11)
plot(x, pH[s1:e1],'r-', x, pH[s2:e2],'g-', x, pH[s3:e3],'b-')
#plot(x, pH_phc[s1_phc:e1_phc],'g-', x, pH_phc[s2_phc:e2_phc],'g-', x, pH_phc[s3_phc:e3_phc],'g-')
plot(x_ms,pH_ms_1[:,1],'r.', x_ms, pH_ms_2[:,1],'g.', x_ms, pH_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(-)')
yticks([4.0, 5.5, 7.0])
title('pH')

subplot(4,3,12)
plot(x, pe[s1:e1],'r-', x, pe[s2:e2],'g-', x, pe[s3:e3],'b-')
#plot(x, pe_phc[s1_phc:e1_phc],'g-', x, pe_phc[s2_phc:e2_phc],'g-', x, pe_phc[s3_phc:e3_phc],'g-')
plot(x_ms,pe_ms_1[:,1],'r.', x_ms, pe_ms_2[:,1],'g.', x_ms, pe_ms_3[:,1],'b.')
xlim([0, 0.4])
xticks([0.1, 0.2, 0.3, 0.4])
ylabel('(-)')
yticks([1.0, 4.5, 8.0])
title('pe')

subplots_adjust(hspace = 0.4, wspace = 0.4)

#ion()
show()
