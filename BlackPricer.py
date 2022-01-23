from scipy.stats import norm
import math

class Black:
    def __init__(self, F_0, K, T, sigma):
        self.F_0 = F_0
        self.K = K
        #self.S = S
        self.T = T
        self.sigma = sigma

    def dPlusBlack(self):
        '''
        Compute the d + term appearing in the Black formula.
        :param F_0:
        :param K:
        :param T:
        :param sigma:
        :return:
        '''

        F_0 = self.F_0
        K = self.K
        T = self.T
        sigma = self.sigma

        d_plus = ((math.log(F_0 / K) + 0.5 * sigma * sigma * T)
                  / sigma / math.sqrt(T))
        return d_plus


    def dMinusBlack(self):
        F_0 = self.F_0
        K = self.K
        T = self.T
        sigma = self.sigma

        d_minus = (Black.dPlusBlack(F_0=F_0, K = K, T=T,
                          sigma=sigma) - sigma * math.sqrt(T))
        return d_minus

    def black(self, isCall):
        '''
        Compute the Black formula.

        :param F_0: forward rate at time t
        :param K: strike
        :param T: expiry
        :param sigma: volatility
        :param isCall:
        :return: option price using Black Scholes
        '''

        F_0 = self.F_0
        K = self.K
        T = self.T
        sigma = self.sigma

        option_value = 0
        if T * sigma == 0.0:
            if isCall:
                option_value = max(F_0 - K, 0.0)
            else:
                option_value = max(K - F_0, 0.0)
        else:
            d1 = Black.dPlusBlack(F_0=F_0, K=K, T = T, sigma = sigma)
            d2 = Black.dMinusBlack(F_0=F_0, K=K, T = T, sigma = sigma)
        if isCall:
            option_value = (F_0 * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            option_value = (K * norm.cdf(-d2) - F_0 * norm.cdf(-d1))
        return option_value



    def computeFirstDerivative(v_u_plus_du , v_u_minus_du , du):
        '''
        Compute the first derivatve of a function using central difference
        @var v_u_plus_du: is the value of the function computed for a positive bump amount du
        @var v_u_minus_du: is the value of the function computed for a negative bump amount du
        @var du: bump amount
        '''
        first_derivative = (v_u_plus_du - v_u_minus_du) / (2.0 * du)
        return first_derivative

    def computeSecondDerivative(v_u, v_u_plus_du, v_u_minus_du, du):
        '''
        Compute the second derivatve of a function using central difference
        @var v_u: is the value of the function
        @var v_u_plus_du: is the value of the function
        computed for a positive bump amount du
        @var v_u_minus_du: is the value of the function
        computed for a negative bump amount du
        @var du: bump amount
        '''
        second_derivative = ((v_u_plus_du - 2.0*v_u + v_u_minus_du)/(du * du))
        return second_derivative


