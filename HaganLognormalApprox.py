import math
from BlackPricer import Black as blk

class SABR:
    def __init__(self, K, T, F_0, alpha_0, beta, nu, rho):
        self.K = K
        self.T = T
        self.F_0 = F_0
        self.alpha_0 = alpha_0
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def haganLogNormalApprox(self):
        '''
        Compute implied vol using Hagan et al. lognormal approximation

        :param K: option strike
        :param T: option expiry in years
        :param F_0: forward interest rate
        :param alph_0: SABR alpha at t = 0
        :param beta: SABR beta
        :param nu: SABR rho
        :param rho: SABR nu
        :return: the Black implied volatility, computed using the Hagan et al. lognormal appoximation
        '''
        K = self.K
        T = self.T
        F_0 = self.F_0
        alpha_0 = self.alpha_0
        beta = self.beta
        nu = self.nu
        rho = self.rho

        one_beta = 1.0 - beta
        one_betasqr = one_beta * one_beta
        if F_0 != K:
            fk = F_0 * K
            fk_beta = math.pow(fk, one_beta / 2.0)
            log_fk = math.log(F_0 / K)
            z = nu / alpha_0 * fk * log_fk
            x = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z + z - rho) /
                          (1-rho)))
            sigma_1 = (alpha_0 / fk_beta / (1.0 + one_betasqr / 24 * log_fk * log_fk +
                                           math.pow(one_beta * log_fk, 4) / 1920.0) * (z / x))
            sigma_exp = (one_betasqr / 24.0 * alpha_0 * alpha_0 / fk_beta / fk_beta + 0.25 * rho * beta * nu * alpha_0/
                         fk_beta + (2.0 - 3.0 * rho * rho)/(24.0 * nu * nu))
            sigma = sigma_1 * (1.0 + sigma_exp * T)
        else:
            f_beta = math.pow(F_0, one_beta)
            f_two_beta = math.pow(F_0, (2.0 - 2.0 * beta))
            sigma = ((alpha_0 / f_beta) * (1.0 + ((one_betasqr / 24.0) *
                                                 (alpha_0 * alpha_0 / f_two_beta)+
                                                 (0.25 * rho * beta * nu * alpha_0 / f_beta) +
                                                 (2.0 - 3.0 * rho * rho)/
                                                 24.0 * nu * nu) * T))
            return sigma

    def computeSABRDelta(self, isCall):
      '''
      Compute Delta using SABR
      '''

      K = self.K
      T = self.T
      F_0 = self.F_0
      alpha_0 = self.alpha_0
      beta = self.beta
      nu = self.nu
      rho = self.rho

      #K, T, F_0, alpha_o, beta, rho, nu,
      small_figure = 1e-6

      F_0_plus_h = F_0 + small_figure
      avg_alpha = (alpha_0 + (rho * nu / math.pow(F_0, beta)) * small_figure)
      vol = SABR.haganLogNormalApprox(K, T, F_0_plus_h, avg_alpha, beta, nu, rho)
      px_f_plus_h = blk.black(F_0_plus_h, K, T, vol, isCall)

      F_0_minus_h = F_0 - small_figure
      avg_alpha = (alpha_0 + (rho * nu / math.pow(F_0, beta)) * (-small_figure))
      vol = SABR.haganLogNormalApprox(K, T, F_0_minus_h, avg_alpha, beta, nu, rho)
      px_f_minus_h = blk.black(F_0_minus_h, K, T, vol, isCall)

      sabr_delta = blk.computeFirstDerivative(px_f_plus_h, px_f_minus_h, small_figure)

      return sabr_delta

