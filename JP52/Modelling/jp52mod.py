#! /usr/bin/env python3

'''
jp52mod

Jardetsky and Press (1952) Rayleigh Wave Coupling to Atmospheric Compression Waves

This file holds a class (jp52base) that can calculate the model proposed by Jardetsky and Press (1952), 
regarding surface waves within a system of atmosphere + two solid subsurface layers (the lower one being a halfspace)

Note that I am taking the coefficients from Eqn 4-257 of EJP57 (i.e., the book) rather than the JP2 paper.
Testing has shown that these coefficients are correct.

One note of caution: I don't think there is an implicit assumption of Poisson's Ratio = 0.25 in the models,
but the model set up below Eqn 36 in JP52 uses such a solid. I have written the program such that it doesn't
make this assumption.
'''

import sys
from numpy import sqrt, sinh, cosh, arange, real, flip, diff, argmin, min, abs, insert, pi
from numpy import asarray, argmax, sign
from numpy import max as npmax
from numpy import copy as npcopy
from numpy import ones as npones
from numpy import size as npsize
import matplotlib.pyplot as plt
from scipy.optimize import newton
from itertools import groupby
import datetime # Just for check plots

# Plot Parameters ------------------------------------------
para = {'axes.labelsize': 16, 'font.size': 16, 'legend.fontsize': 14,
    'xtick.labelsize': 16,'ytick.labelsize': 16, 'figure.subplot.left': 0.12,
    'figure.subplot.right': 0.98, 'figure.subplot.bottom': 0.11,
    'figure.subplot.top': 0.97}
plt.rcParams.update(para)
#-----------------------------------------------------------

def freq_eqn(kH,l0,l1,l2,l3,l4,n1,n3):
    '''
    The frequency equation of JP52 (Eqn 36)
    '''
    t0 = l0
    t1 = l1*sinh(n1*kH)*sinh(n3*kH)
    t2 = l2*sinh(n1*kH)*cosh(n3*kH)
    t3 = l3*cosh(n1*kH)*sinh(n3*kH)
    t4 = l4*cosh(n1*kH)*cosh(n3*kH)
    return t0+t1+t2+t3+t4

class jp52base():
    def __init__(self,inputdict={'alpha0':None, 'rho0':None,
                                 'alpha1':None, 'beta1':None, 'rho1':None,
                                 'alpha2':None, 'beta2':None, 'rho2':None}):
        '''
        JP52 appear to set up the model such that velocities are calculated from the
        Lame Parameters.

        As a reminder:
        beta = sqrt(mu/rho)
        lambda = sqrt((lambda+2*mu)/rho)
        '''

        # Check to make sure there are no None values in the inputdict:
        if any([v==None for v in inputdict.values()]):
            print('At least one None value in the input parameter dictionary. Exiting')
            sys.exit(1)

        # test all params available
        tp = ['alpha0','rho0','alpha1','beta1','rho1','alpha2','beta2','rho2']
        for param in tp:
            try:
                print(param,'=',inputdict[param])
            except:
                print(param,' is missing from input')
                sys.exit(1)

        self.paramdict = inputdict

        return

    def lame_parameters(self,printflag=False):
        '''
        We're going to calculate the lame parameters FROM the vp,vs values.
        '''
        a1sq = self.paramdict['alpha1']*self.paramdict['alpha1']
        a2sq = self.paramdict['alpha2']*self.paramdict['alpha2']
        b1sq = self.paramdict['beta1']*self.paramdict['beta1']
        b2sq = self.paramdict['beta2']*self.paramdict['beta2']

        self.mu1 = self.paramdict['rho1']*b1sq
        self.mu2 = self.paramdict['rho2']*b2sq

        self.lambda1 = (self.paramdict['rho1']*a1sq)-(2*self.mu1)
        self.lambda2 = (self.paramdict['rho2']*a2sq)-(2*self.mu1)

        self.poissonsratio1 = (a1sq-(2*b1sq))/(2*(a1sq-b1sq))
        self.poissonsratio2 = (a2sq-(2*b2sq))/(2*(a2sq-b2sq))

        if printflag:
            print('mu1: ',self.mu1 )
            print('lambda1: ',self.lambda1 )
            print('poissonsratio1: ',self.poissonsratio1 )
            print('mu2: ',self.mu2 )
            print('lambda2: ',self.lambda2 )
            print('poissonsratio2: ',self.poissonsratio2 )

        return


    def V_and_n_coeffs(self,c):
        '''
        Calculate the V and n coeffs - hopefully explanatory from the EJP57 formulation.
        '''
        self.V = c/self.paramdict['beta1']
        self.V2 = self.V*self.V
        self.V4 = self.V2*self.V2
        self.V8 = self.V4*self.V4

        self.n0 = sqrt(1.-((self.V2*self.paramdict['beta1']*self.paramdict['beta1'])/(self.paramdict['alpha0']*self.paramdict['alpha0']))+0j)
        self.n1 = sqrt(1.-((self.V2*self.paramdict['beta1']*self.paramdict['beta1'])/(self.paramdict['alpha1']*self.paramdict['alpha1']))+0j)
        self.n2 = sqrt(1.-((self.V2*self.paramdict['beta1']*self.paramdict['beta1'])/(self.paramdict['alpha2']*self.paramdict['alpha2']))+0j)
        self.n3 = sqrt(1.-(self.V2)+0j)
        self.n4 = sqrt(1.-((self.V2*self.paramdict['beta1']*self.paramdict['beta1'])/(self.paramdict['beta2']*self.paramdict['beta2']))+0j)

        return
    
    def WXYZ_coeffs(self):
        '''
        Calculate the X, Y, Z, W coefficients (eqn 33 of JP52 paper)
        '''
        self.X = ((self.paramdict['rho2']/self.paramdict['rho1'])*self.V*self.V)-(2*((self.mu2/self.mu1)-1))
        self.W = 2*((self.mu2/self.mu1)-1)
        self.Y = (self.V2)+self.W
        self.Z = self.X-(self.V2)

        return

    def G_coeffs(self,pubflag='EJP57'):
        '''
        The G coefficients (i.e., Eqn34 of JP52).

        This is the one equation where there is a difference between the paper and the book (EJP57)

        After testing, it appears that the formulation in EJP57 (i.e., their Eqn 4-257) is the corrected
        version - using the version in JP52 doesn't converge correctly.

        Added a flag, that as default takes the paper. Can also be 'JP52' to show the INCORRECT version.
        Always keep as EJP57 if undertaking a modelling run that you wish to be correct!
        '''

        if pubflag == 'JP52':
            print('WARNING: Use with Caution. Unlikely to be correct.')
            self.G1 = (self.X*self.Y)-(self.n2*self.n4*self.W*self.Y)
        elif pubflag == 'EJP57':
            self.G1 = (self.X*self.Z)-(self.n2*self.n4*self.W*self.Y)
        else:
            print('pubflag is wrong. Should be: JP52 or EJP57')

        self.G2 = (self.Z*self.Z)-(self.n2*self.n4*self.Y*self.Y)
        self.G3 = (self.n2*self.n4*self.W*self.W)-(self.X*self.X)

        return
    
    def l_coeffs(self):
        '''
        The l coeffs (Eqn 35 of JP52)
        '''

        self.l0 = 4*(2.-self.V2)*self.G1

        self.l1 = (((2.-self.V2)**2)*(1/(self.n1*self.n3))*self.G2 -
                    4*self.n1*self.n3*self.G3 -
                    (((self.paramdict['rho0']*self.paramdict['rho2'])/(self.paramdict['rho1']*self.paramdict['rho1']))*
                    ((self.n1*self.n4)/(self.n0*self.n3))*self.V8))

        self.l2 = (((-1*(2.-self.V2)**2)*(self.paramdict['rho2']/self.paramdict['rho1'])*(self.n2/self.n1)*self.V4) +
                    ((4*self.paramdict['rho2']*self.n1*self.n4*self.V2)/self.paramdict['rho1']) +
                    ((self.paramdict['rho0']*self.n1*self.V4*self.G3)/(self.paramdict['rho1']*self.n0)))

        self.l3 = (((-1*(2.-self.V2)**2)*(self.paramdict['rho2']/self.paramdict['rho1'])*(self.n4/self.n3)*self.V4) +
                    ((4*self.paramdict['rho2']*self.n2*self.n3*self.V4)/self.paramdict['rho1']) +
                    ((self.paramdict['rho0']*self.V4*self.G2)/(self.paramdict['rho1']*self.n0*self.n3)))

        self.l4 = (((2.-self.V2)**2)*self.G3 - 4*self.G2 - 
                    ((self.paramdict['rho0']*self.paramdict['rho2'])/(self.paramdict['rho1']*self.paramdict['rho1']))*(self.n2/self.n0)*self.V8)
            

    def calc_coeffs(self,c,pubflag='EJP57',printlame=False):
        '''
        Ordered recalculation of coefficients for new phase velocity, c
        '''
        self.V_and_n_coeffs(c)
        self.lame_parameters(printflag=printlame)
        self.WXYZ_coeffs()
        self.G_coeffs(pubflag=pubflag)
        self.l_coeffs()

        return

    def kHrootfind(self,kH0):
        '''
        With the calculated l coefficients, solve JP52 eqn 36 to find the roots (kH)
        that correspond to solutions - only one??

        Attempting to use the Newton solver (as the frequency eqn is complex).
        In the nomenclature of scipy.optimize.newton x = kH for our purposes.
        Requires an x0 value (i.e., an estimated value of kH
        thought to be close to the zero) to search from. Initialising this can be
        problematic
        '''

        try:
            root = newton(freq_eqn,kH0,args=(self.l0,self.l1,self.l2,self.l3,self.l4,self.n1,self.n3))
            if real(root)<0:
                root = None # Trying to stop the negative root nonsense at source.
        except RuntimeError:
            root = None

        return root

    def return_phase_vel_curves(self,asymptote=0.924,pltflag=False,figfile=None):
        '''
        The phase velocity curves (i.e., c/beta1 in Fig. 1 of JP52) need to be returned
        before the group velocity curves (i.e., U/beta1 in Fig. 1 of JP52) can be calculated via
        numerical differentiation.

        The difficulty is that there is a discontinuity in the vicinity of c/beta1 = alpha0/beta1.

        Therefore we need to return two curves. One 'above' the discontinuity (associated with
        Branch 1 returns) and one 'below' the discontinuity (associated with Branch 2 returns)

        asymptote (=0.924) this is the value that c/beta1 asymptotes to at high values of kH.
        I've found this to be very sensitive (and will cause the program to fail at times, due to
        a lack of convergence)
        '''


        # I realise this param is not really necessary; just adding for ease of comparison with paper.
        critical_c_ratio = self.paramdict['alpha0']/self.paramdict['beta1'] # Critical alpha/beta ratio.

        # ---
        # The root finding for branch I is occassionally ill-behaved. I have tried to add
        # a number of conditions to 'catch' these issues.
        print('Looking for roots for Branch I')
        upperlim = self.paramdict['beta1']*2.8 # A rather arbitrary upper limit, but something to aim for...
        cvec_in = arange(critical_c_ratio*self.paramdict['beta1']*1.0005,upperlim,0.02)
        kHroot_I = []
        cvec_I = []
        kH0 = 2.0 # Will this work at all times??
        for cval in cvec_in:
            #print('cval, branch I: {}, kH0: {}'.format(cval,kH0)) # Debug statement
            self.calc_coeffs(c=cval,printlame=False,pubflag='EJP57')
            kHroot = self.kHrootfind(kH0=kH0)
            #print('cval, branch I: {}, kH0: {}, kHroot {}'.format(cval,kH0, kHroot)) # Debug statement
            if (kHroot != None): # i.e., if there is a kH root value
                if real(kHroot)<1e-3:
                    # Attempt to stop loop if kH goes too close to zero (as we approach it from more positive nos.)
                    # Note that negative real parts of the root will trigger root to become None. However, this break
                    # clause stops the branch kH going negative, returning Nones, but then returning nonsensical
                    # positive kH at larger c_vecs
                    print('Breaking Out, real(kHroot)={}'.format(real(kHroot)))
                    break

                cvec_I.append(cval) # Add the phase velocity (cval) to the extended vector
                kHroot_I.append(kHroot) # Add the root to the extended vector
                kH0 = real(kHroot_I[-1]) # So as I progress, I update KH0 to be the value calculated at the previous point

        # For a well-behaved Branch I, I expect the gradient of dC/d(kH) to be negative.
        # Now want to go through, and remove any portion at the start where the gradient of dC/d(kH) is positive
        # Also want to travel along branch and remove any portions beyond which the gradient goes positive.
        # Note that in the program we travel downwards in kH in the vector....

        # We achieve the above by identifying the longest portion of the kHvector with continuous
        # negative values
        dkHsign = sign(diff(real(kHroot_I)))
        # maxpattern is the max length portion of the vector where values are continuously negative
        maxpattern = max((list(g) for k, g in groupby(dkHsign, key=lambda i: i < 0)), key=len)
        # Now identify where the 'max' list is in dkHsign - assuming only one occurrence of pattern!
        for i in range(len(dkHsign)):
            if ((dkHsign[i] == maxpattern[0]) and (dkHsign[i:i+len(maxpattern)] == maxpattern)).all():
                imax = i
                break

        kHroot_I = kHroot_I[imax:imax+len(maxpattern)] # Take the kHroot as the longest negative continuous section
        cvec_I = cvec_I[imax:imax+len(maxpattern)] # Take the corresponding cvec section

        # For ease of interpretation, flipping the vectors so that kH runs in ascending order
        cvec_I = flip(cvec_I)
        kHroot_I = flip(kHroot_I)

        '''
        Now moving on to find the roots of Branch II
        '''

        print('Looking for roots for Branch II')
        cvec_in = arange(critical_c_ratio*self.paramdict['beta1']*0.9999,self.paramdict['beta1']*asymptote,-0.02)
        kHroot_II = []
        cvec_II = []
        kH0 = kHroot_I[-1] #  [Here taking the end kH0 from branch I, it is already flipped]
        print('kHO at start of branch II: {}'.format(kH0))
        for cval in cvec_in:
            #print('cval, branch II: {}, kH0: {}'.format(cval,kH0)) # Debug statement
            self.calc_coeffs(c=cval,printlame=False,pubflag='EJP57')
            kHroot = self.kHrootfind(kH0=kH0)
            if (kHroot != None):
                if (real(kHroot) > 0.0):
                    cvec_II.append(cval)
                    kHroot_II.append(kHroot)
                    kH0 = real(kHroot_II[-1]) # So as I progress, I update KH0 to be the value calculated at the previous point     
        # For ease of interpretation, flip so that 
        if (cvec_II[-1]<cvec_II[0]):
            cvec_II = flip(cvec_II)
            kHroot_II = flip(kHroot_II)

        #print('ROOTS:',kHroot_I[0],kHroot_I[-1],kHroot_II[0],kHroot_II[-1])

        self.cdict = {'cvec_I':cvec_I,'kH_I':real(kHroot_I),'cvec_II':cvec_II,'kH_II':real(kHroot_II)}

        if pltflag:
            fig = plt.figure(figsize=(4.0,7.0))
            ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
            ax.plot(real(self.cdict['kH_I']),self.cdict['cvec_I']/self.paramdict['beta1'],'k-')
            ax.plot(real(self.cdict['kH_II']),self.cdict['cvec_II']/self.paramdict['beta1'],'b-')

            restrictx = False # Internally debug flag
            if restrictx:
                ax.set_xlim(self.cdict['kH_I'][-1]*asarray([0.9,1.1]))
            else:
                ax.set_xlim([0,9])
                ax.set_xticks([0,1,2,3,4,5,6,7,8,9])

            ax.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8])
            ax.set_ylim([0.4,2.9])
            ax.set_xlabel('kH')
            ax.set_ylabel(r'c/$\beta_1$')
            ax.grid('on')
            if figfile is None:
                plt.show()
            else:
                fig.savefig(figfile,bbox_inches='tight')
            plt.close()


        return

    def return_group_vel_curves(self,pltflag=False):
        '''
        Given a set of phase velocity curves, calculate the group velocity curves via numerical differentiation...
        '''

        try:
            self.gdict = {}

            # --- Calculate Group Velocity Curve for Branch I
            kH_I,gvecbeta1rat_I, cvec_I = self.calc_gvel(self.cdict['kH_I'],self.cdict['cvec_I'])

            # Going to trim the branch I, so that it finishes at the minima, rather than has the start of Branch III
            # This is in some ways an artificiality, but I want to make sure I have branch output that can then be
            # compared to the recorded data (for Evie Read's Foulness project)
            # The following parameters are returned from trim_I:
            # self.gdict['kH_I'],self.gdict['gvecbeta1rat_I']
            self.trim_I(kH_I,gvecbeta1rat_I, cvec_I)

            # --- Calculate the branches that will become branch II and branch III
            kH_II_III,gvecbeta1rat_II_III, cvec_II_III = self.calc_gvel(self.cdict['kH_II'],self.cdict['cvec_II'])

            # Branch II currently contains information about branch II and branch III. Would like to split
            # Following variables are calculated in split_II_III:
            # self.gdict['KH_II'],self.gdict['gvecbeta1rat_II'],self.gdict['KH_III'],self.gdict['gvecbeta1rat_III']
            self.split_II_III(kH_II_III,gvecbeta1rat_II_III,cvec_II_III,testplot=False)
            #--- Now want to add the part of the original Branch I data (kH_I,gvecbeta1rat_I) that has
            # kH values larger than that at the gvecbeta1rat minima to Branch II, to ensure that we are not
            # missing a frequency section for comparison with the data. Just need enough points that I can fit a smooth
            # (spline?) through the data points.
            self.extend_II(kH_I,gvecbeta1rat_I,cvec_I,testplot=False)

        except NameError:
            print('No phase velocity curves calculated')
            raise

                
        if pltflag:
            fig = plt.figure(figsize=(4.0,7.0))
            ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
            ax.plot(real(self.cdict['kH_I']),self.cdict['cvec_I']/self.paramdict['beta1'],'k-')
            ax.plot(real(self.cdict['kH_II']),self.cdict['cvec_II']/self.paramdict['beta1'],'b-')
            ax.plot(real(self.cdict['kH_III']),self.cdict['cvec_III']/self.paramdict['beta1'],'r-')
            ax.plot(real(self.gdict['kH_I']),self.gdict['gvecbeta1rat_I'],'k-',lw=4)
            ax.plot(real(self.gdict['kH_II']),self.gdict['gvecbeta1rat_II'],'b-',lw=4)
            ax.plot(real(self.gdict['kH_III']),self.gdict['gvecbeta1rat_III'],'r-',lw=4)
            ax.set_xlim([0,9])
            ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
            ax.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8])
            ax.set_ylim([0.35,2.9])
            ax.set_xlabel('kH')
            ax.set_ylabel(r'c/$\beta_1$, U/$\beta_1$')
            ax.grid('on')
            plt.show()

            plt.close()

    def calc_gvel(self,kHvec,cvec):
        '''
        Calculated the group velocity curve via numerical differentiation.
        
        Initial attempt uses dimensionless units. Is this correct??
        '''

        print('*** calc_gvel ***')

        cb_rat = cvec/self.paramdict['beta1']
        
        kHdiff = diff(kHvec)
        diffvec = (cb_rat[2:]-cb_rat[:-2])/(kHdiff[:-1]+kHdiff[1:])
        gvecbeta1rat = cb_rat[1:-1] + kHvec[1:-1]*diffvec

        testplot = False # Plot to look at the change in gvec over the branch.
        if testplot:
            plt.semilogy(kHvec[1:-1],gvecbeta1rat,'ko-')
            plt.xlabel('kH')
            plt.ylabel('gvecbeta1rat')
            plt.savefig('temp'+datetime.datetime.now().strftime('%s%f')+'.png')
            plt.close()

        # Now want the phase velocities output at the same vector length...
        return kHvec[1:-1], gvecbeta1rat, cvec[1:-1]

    def split_II_III(self,kH,gvecbeta1rat,cvec,testplot=False):
        '''
        Expects vectors of kH and U/beta1 values that contain both branch II and branch III information

        Also the phase velocity vector (cvec) output from the group velocity calcs.

        It is ASSUMED that there is a minimum in gvecbeta1rat, and that values associated with lower kH are
        Branch III, and those associated with higher kH are Branch II.
        '''

        # Make sure that the kH vectors are oriented low to high
        if kH[0]>kH[-1]:
            kH = flip(kH)
            gvecbeta1rat = flip(gvecbeta1rat)
            cvec = flip(cvec)

        imin = argmin(gvecbeta1rat)
        self.gdict['kH_III'] = kH[:imin]
        self.gdict['kH_II'] = kH[imin:]
        self.gdict['gvecbeta1rat_III'] = gvecbeta1rat[:imin]
        self.gdict['gvecbeta1rat_II'] = gvecbeta1rat[imin:]
        self.cdict['kH_III'] = kH[:imin]
        self.cdict['kH_II'] = kH[imin:]
        self.cdict['cvec_III'] = cvec[:imin]
        self.cdict['cvec_II'] = cvec[imin:]

        print('IIcheck1:',npsize(self.cdict['cvec_II']),npsize(self.gdict['kH_II']))

        if self.gdict['kH_II'][0] > self.gdict['kH_II'][-1]:
            self.gdict['kH_II'] = flip(self.gdict['kH_II'])
            self.gdict['gvecbeta1rat_II'] = flip(self.gdict['gvecbeta1rat_II'])
            self.cdict['kH_II'] = flip(self.cdict['kH_II'])
            self.cdict['cvec_II'] = flip(self.cdict['cvec_II'])
        
        if self.gdict['kH_III'][0] > self.gdict['kH_III'][-1]:
            self.gdict['kH_III'] = flip(self.gdict['kH_III'])
            self.gdict['gvecbeta1rat_III'] = flip(self.gdict['gvecbeta1rat_III'])
            self.cdict['kH_III'] = flip(self.cdict['kH_III'])
            self.cdict['cvec_III'] = flip(self.cdict['cvec_III'])
        

        if testplot:
            plt.close()
            plt.plot(self.gdict['kH_II'],self.gdict['gvecbeta1rat_II'],'r-')
            plt.plot(self.gdict['kH_III'],self.gdict['gvecbeta1rat_III'],'b-')
            plt.title('Split_II_III_test_plot')
            plt.show()
            plt.close()

        return

    def trim_I(self,kH,gvecbeta1rat,cvec):
        '''
        Expects input vectors of kH and U/beta1 (for branch I)
        Want to trim all values at kH values above the U/beta1 minimum.
        This is to help with the data/model fitting procedure.
        '''
        # Make sure that the kH vectors are oriented low to high
        if kH[0]>kH[-1]:
            kH = flip(kH)
            gvecbeta1rat = flip(kH)

        imin = argmin(gvecbeta1rat)
        self.gdict['kH_I'] = kH[:imin+1]
        self.gdict['gvecbeta1rat_I'] = gvecbeta1rat[:imin+1]
        self.cdict['kH_I'] = kH[:imin+1]
        self.cdict['cvec_I'] = cvec[:imin+1]

        return

    def extend_II(self,kH,gvecbeta1rat,cvec,testplot=False):
        '''
        Using the branch I model points from kH values above the gvecbeta1rat minima to extend
        branch II.

        kH and gvecbeta1rat are the values for branch 1
        cvec should be the branch I phase velocity vector
        '''

        print('0 len check:',len(self.cdict['kH_II']),len(self.cdict['cvec_II']))

        # Make sure that the (branch I) kH vectors are oriented low to high
        if kH[0]>kH[-1]:
            kH = flip(kH)
            gvecbeta1rat = flip(kH)
            cvec = flip(cvec)

        # Find the index related to the gvecbeta1rat minima
        imin = argmin(gvecbeta1rat)
        # Find the index related to the Branch III minimum kH, and then calculate the point
        # that is 20% of the gap between this and the Branch II minimum. We don't want to 
        # go to higher kH values (as it is likely that Branch I will turn into Branch III at
        # about this kH value?)
        kHIImin = min(self.gdict['kH_II'])
        kHIIImin = min(self.gdict['kH_III'])
        kHmidpoint = kHIIImin+(0.2*(kHIImin-kHIIImin))
        imax = argmin(abs(kH-kHmidpoint)) # i.e., the maximum index for kH (below kHIIImin)

        if testplot:
            plt.plot(self.gdict['kH_III'],self.gdict['gvecbeta1rat_III'],'r-')
            plt.plot(self.gdict['kH_I'],self.gdict['gvecbeta1rat_I'],'k-')
            plt.plot(self.gdict['kH_II'],self.gdict['gvecbeta1rat_II'],'b-')
            plt.plot(kH[imin:imax+1],gvecbeta1rat[imin:imax+1],'go-')
            plt.title('extend_II testplot')
            plt.show()

        print(kH[imin:imax+1][-5:])
        print(self.gdict['kH_II'][:5])

        self.gdict['kH_II'] = insert(self.gdict['kH_II'],0,kH[imin:imax+1])
        self.gdict['gvecbeta1rat_II'] = insert(self.gdict['gvecbeta1rat_II'],0,gvecbeta1rat[imin:imax+1])
        self.cdict['kH_II'] = insert(self.cdict['kH_II'],0,kH[imin:imax+1])
        self.cdict['cvec_II'] = insert(self.cdict['cvec_II'],0,cvec[imin:imax+1])

        print('len check:',len(self.cdict['kH_II']),len(self.cdict['cvec_II']))

        if testplot:
            plt.plot(self.gdict['kH_II'],'k.')
            plt.show()

        return

    def gval_into_physical_units(self,pltflag=False):
        '''
        All the calculations, and model curve generation, are done in the
        dimensionless form of JP52. Now want to ensure that we convert to
        physical units - as want ultimately to compare to the data (which
        will be provided as frequency vs group velocity, per branch)
        '''

        self.gdict['gvec_I'] = self.gdict['gvecbeta1rat_I']*self.paramdict['beta1']
        self.gdict['gvec_II'] = self.gdict['gvecbeta1rat_II']*self.paramdict['beta1']
        self.gdict['gvec_III'] = self.gdict['gvecbeta1rat_III']*self.paramdict['beta1']

        # To get f I need to use the equation on p193 of EJP57 that relates period to
        # the PHASE VELOCITY, the shear-wave speed in the top layer, and the kH parameter.
        # This means I am going to have to chase the PHASE VELOCITIES through the program
        # and ensure they are consistent with the group velocity vectors.
        
        self.gdict['f_I'] = self.freq_from_kH_H(branch='I')
        self.gdict['f_II'] = self.freq_from_kH_H(branch='II')
        self.gdict['f_III'] = self.freq_from_kH_H(branch='III')

        if pltflag:
            fig = plt.figure(figsize=(4.0,7.0))
            ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
            ax.plot(real(self.gdict['f_I']),self.gdict['gvec_I'],'k-',lw=4)
            ax.plot(real(self.gdict['f_II']),self.gdict['gvec_II'],'b-',lw=4)
            ax.plot(real(self.gdict['f_III']),self.gdict['gvec_III'],'r-',lw=4)
            ax.set_xlabel('f (Hz)')
            ax.set_ylabel(r'U (km/s)')
            ax.grid('on')
            plt.show()
            plt.close()


        return

    def freq_from_kH_H(self,branch='I'):
        '''
        Return frequency vector (in physical units) - using Eqn at base of p193 of EJP57
        '''
        return (((self.cdict['cvec_'+branch]/self.paramdict['beta1'])*self.gdict['kH_'+branch])/
                                (2*pi*self.paramdict['H']/self.paramdict['beta1']))

 
    def overview_plot(self,df_freqbranch=None,figfile=None):
        '''
        Provide an overview plot, of the arrivals and the velocity model.

        Note I'm adding df_freqbranch such that in the future I can easily
        add the ability to plot data on top of the model as required.
        '''

        fig = plt.figure(figsize=(9.5,6.0))
        ax1 = fig.add_axes([0.15, 0.15, 0.45, 0.80])
        ax2 = fig.add_axes([0.65, 0.45, 0.33, 0.50])
        ax3 = fig.add_axes([0.65,0.15,0.33,0.25])

        # ax1 = the group velocity curves
        ax1.plot(real(self.gdict['f_I']),self.gdict['gvec_I'],'k-',lw=4,label='branch I')
        ax1.plot(real(self.gdict['f_II']),self.gdict['gvec_II'],'b-',lw=4,label='branch II')
        ax1.plot(real(self.gdict['f_III']),self.gdict['gvec_III'],'r-',lw=4,label='branch III')
        ax1.set_xlabel('f (Hz)')
        ax1.set_ylabel(r'U (m/s)')
        ax1.legend(loc='upper right')
        ax1.grid('on')

        # ax2 = the model
        # Ground surface / layer
        ax2.plot([0.85*self.paramdict['beta1'],2*self.paramdict['beta2']],[0,0],'-',color='silver')
        ax2.plot([0.85*self.paramdict['beta1'],2*self.paramdict['beta2']],-1*self.paramdict['H']*npones(2),'-',color='silver')

        # Shear Wave Velocity profile
        ax2.plot(self.paramdict['beta1']*npones(2),[0,-1*self.paramdict['H']],'k-',lw=2)
        ax2.plot([self.paramdict['beta1'],self.paramdict['beta2']],-1*self.paramdict['H']*npones(2),'k-',lw=2)
        ax2.plot(self.paramdict['beta2']*npones(2),[-1*self.paramdict['H'],-2*self.paramdict['H']],'k-',lw=2)

        ax2.plot(self.paramdict['alpha0']*npones(2),[0,self.paramdict['H']],'k--',lw=2)

        ax2.set_ylim([-2*self.paramdict['H'],self.paramdict['H']])
        ax2.set_xlim([0.85*self.paramdict['beta1'],2*self.paramdict['beta2']])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.text(1.95*self.paramdict['beta2'],0.5*self.paramdict['H'],r'$\alpha_0$={}m/s'.format(int(self.paramdict['alpha0'])),va='center',ha='right')
        ax2.text(1.95*self.paramdict['beta2'],-0.5*self.paramdict['H'],
                r'$\beta_1$={}m/s'.format(int(self.paramdict['beta1']))+'\n'+'H={}m'.format(self.paramdict['H']),va='center',ha='right')
        ax2.text(1.95*self.paramdict['beta2'],-1.5*self.paramdict['H'],r'$\beta_2$={}m/s'.format(int(self.paramdict['beta2'])),va='center',ha='right')

        # Print Out the input dictionary values
        ax3.text(0.05,0.85,r'$\alpha_0$: {}m/s'.format(int(self.paramdict['alpha0'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.05,0.7,r'$\rho_0$: {}kg/m$^3$'.format(self.paramdict['rho0']),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.05,0.55,r'$\alpha_1$: {}m/s'.format(int(self.paramdict['alpha1'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.05,0.4,r'$\beta_1$: {}m/s'.format(int(self.paramdict['beta1'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.05,0.25,r'$\rho_1$: {}kg/m$^3$'.format(int(self.paramdict['rho1'])),transform=ax3.transAxes,ha='left',va='center',size=10)

        ax3.text(0.55,0.85,r'$\alpha_2$: {}m/s'.format(int(self.paramdict['alpha2'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.55,0.7,r'$\beta_2$: {}m/s'.format(int(self.paramdict['beta2'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.55,0.55,r'$\rho_2$: {}kg/m$^3$'.format(int(self.paramdict['rho2'])),transform=ax3.transAxes,ha='left',va='center',size=10)
        ax3.text(0.55,0.4,r'H: {}m'.format(int(self.paramdict['H'])),transform=ax3.transAxes,ha='left',va='center',size=10)  
        ax3.axis('off')

        if figfile == None:
            plt.show()
            plt.close()
        else:
            fig.savefig(figfile,bbox_inches='tight')
            plt.close()

if __name__ == '__main__':

    '''
    Test set up of model
    '''

    # Using parameters from pg.140 of JP52
    ft_to_m = 0.3048

    # Set-up the parameter dictionary
    # Required params: alpha0,rho0,alpha1,beta1,rho1,alpha2,beta2,rho2
    # Here I have to calculate some of the parameters from the Lame parameters (and
    # Poisson ratios) given in the text
    paramdict = {'alpha0':1070*ft_to_m,
                 'beta1':800*ft_to_m,
                 'rho0':1.293}
    paramdict['rho1'] = paramdict['rho0']*1000.
    mu1mu2rat = 13.77
    rho2rho1rat = 1.39
    poissonratio = 0.25
    paramdict['rho2'] = rho2rho1rat*paramdict['rho1']
    mu1 = paramdict['beta1']*paramdict['beta1']*paramdict['rho1']
    mu2 = mu1*mu1mu2rat
    print('mu1/mu2:',mu1,mu2)
    paramdict['beta2'] = sqrt(mu2/paramdict['rho2'])
    paramdict['alpha1'] = sqrt((2*(poissonratio-1))/(2*poissonratio-1))*paramdict['beta1']
    paramdict['alpha2'] = sqrt((2*(poissonratio-1))/(2*poissonratio-1))*paramdict['beta2']
    paramdict['H'] = 90. # This is NOT in the original JP52 model, but required to give results in physical units.

    print('paramdict: ',paramdict)

    # Sets up the model with the parameter dictionary
    jp52test = jp52base(inputdict=paramdict)
    # Returns the phase velocity curves (can plot if you wish)
    jp52test.return_phase_vel_curves(pltflag=False,asymptote=0.924)
    # Returns the group velocity curves (non-dimensionalised)
    # If pltflag = True, plots the curves to screen
    jp52test.return_group_vel_curves(pltflag=False)
    # Now need function to return a frequency vector and a group velocity vector in
    # 'real' dimensions -> will require a layer thickness (H)
    # If pltflag = True, plots the curves to screen
    jp52test.gval_into_physical_units(pltflag=False)

    jp52test.overview_plot(figfile='temp.png')

