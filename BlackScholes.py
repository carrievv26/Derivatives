import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

class BSVanillaMC:

    def __init__(self, callput, T, S, K, sigma, r, NPaths, NSteps, useAntithetic):
        '''
        :param callput: call if value is 1, put if value is -1
        :param S: spot price
        :param K: strike price
        :param T: expiry (in years)
        :param sigma: BSM implied volatility
        :param r: discounting rates
        :param NPaths: number of simulation paths
        :param NSteps: number of simulation steps
        :param useAntithetic: True or False
        '''

        self.callput = callput
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.NPaths = NPaths
        self.NSteps = NSteps
        self.useAntithetic = useAntithetic

    def d(plusminus, t, s, sigma, r):
        '''
        Compute the d1 term (d+) in the Black Scholes Merton Formula
        :param plusminus:
        :param t: period time
        :param s: spot
        :param sigma: volatility
        :param r: discounting rate
        :return: if +: probability of S > K otherwise S < K

        '''

        if plusminus == "+":
            d = (math.log(s)+(r+0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
        else:
            d = (math.log(s)+(r-0.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
        return d


    def BSPrice(self):
        '''
        Pricing based on Black Scholes fomula

        :return: analytical price based on BS
        '''
        callput = self.callput
        S = self.S
        K = self.K
        T = self.T
        sigma = self.sigma
        r = self.r

        df = math.exp(-r * T)
        sigma_sqrt = sigma * math.sqrt(T)
        #
        d1 = BSVanillaMC.d("+", T, (S / K), sigma, r)
        d2 = d1 - sigma_sqrt
        F = S * math.exp(r * T)
        price = callput * df * (F * norm.cdf(d1 * callput) - K * norm.cdf(d2 * callput))
        return (price)

    def BSMCPrice(self):
        '''
        Pricing using Monte Carlo based on Black Scholes fomula

        :return: option price using Monte Carlo simulation method based on Black Scholes
        '''

        updown = self.callput
        S = self.S
        K = self.K
        T = self.T
        sigma = self.sigma
        r = self.r
        NPaths = self.NPaths
        NSteps = self.NSteps
        useAntithetic = self.useAntithetic

        ts = np.linspace(0, T, NSteps)

        spots = S * np.ones((NSteps, NPaths))
        spotsb = S * np.ones((NSteps, NPaths))
        gaussianvars = np.zeros((NSteps, NPaths))
        dts = np.zeros(NSteps - 1)
        payoff = np.zeros(NPaths)
        vars_payoff = np.zeros(NPaths)

        for i in range(NSteps-1):
            gaussianvars[i,:] = np.random.normal(0,1,NPaths)
            dts[i] = ts[i+1] - ts[i]

        sigma2 = sigma**2
        # print (gaussianvars[NSteps-1,:])
        for i in range(NSteps-1):
            dti = dts[i]
            for j in range(NPaths):
                spots[i+1,j] = spots[i,j] * math.exp((r-0.5*sigma2)*dti+sigma*math.sqrt(dti)*gaussianvars[i,j])

                if useAntithetic:
                    spotsb[i+1,j] = spotsb[i,j] * math.exp((r-0.5*sigma2)*dti-sigma*math.sqrt(dti)*gaussianvars[i,j])
        if useAntithetic:
            for j in range(NPaths):
                payoff[j] = 0.5*(max(updown*(spots[NSteps-1,j]-K),0) + max(updown*(spotsb[NSteps-1,j]-K),0))
        else:
            for j in range(NPaths):
                payoff[j] = max(updown*(spots[NSteps-1,j]-K),0)

        for j in range(NPaths):
            vars_payoff[j] = (payoff[j]-np.mean(payoff))*(payoff[j]-np.mean(payoff))

        pv = (math.exp(-r*T) * payoff)/NPaths
        sigsqrtn = (math.sqrt(np.sum(vars_payoff*math.exp(-2*r*T)/(NPaths-1))/math.sqrt(NPaths)))
        i1 = np.sum(pv) - 1.96*sigsqrtn
        i2 = np.sum(pv) + 1.96*sigsqrtn

        if (useAntithetic == True):
            tag = "Antithetic variables"
        else:
            tag = "No antithetic variables"
        print("--------Price mode:", tag)
        print("lower boundary: ", i1)
        print("higher boundary: ", i2)

        plt.plot(ts, spots[:,0],'-', ts, spots[:,1],'-')
        plt.show()
        return np.sum(pv)

if __name__ == '__main__':


    analyticalPrice = BSVanillaMC(1, 1, 100, 96, 0.15, 0.05, 2000, 2, True).BSPrice()
    print('Analytical price: ', analyticalPrice)

    MCPrice = BSVanillaMC(1, 1, 100, 96, 0.15, 0.05, 2000, 2, False).BSMCPrice()
    print("MC price w/o Antithetic: ", MCPrice)

    MCPrice = BSVanillaMC(1, 1, 100, 96, 0.15, 0.05, 2000, 2, True).BSMCPrice()
    print("MC price with Antithetic: ", MCPrice)
    
    

    def localVol_forwardVol(vol1, vol2, T1, T2):
        '''

        :param vol1: implied volatility with expiry T1
        :param vol2: implied volatility with expiry T2
        :param T1: expiry T1 (in years)
        :param T2: expiry T2 (in years), usually, T2 > T1
        :return: forward IV with expiry T2-T1 in T1
        '''
        vol1 = vol1
        vol2 = vol2
        T1 = T1
        T2 = T2

        forwardVol = math.sqrt((vol2**2*T2 - vol1**2*T1) / (T2-T1))
        return print("Forward IV for " + str(T2-T1) + "y expiry in " + str(T1) + "y time: ", forwardVol)

    localVol_forwardVol(0.1, 0.15, 1, 3)
