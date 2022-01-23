import math

def hanganLogNormalApprox(K, T, F_0, alpha_0, beta, nu, rho):
    '''

    :param K: option strike
    :param T: option expiry in years
    :param F_0: forward interest rate
    :param alph_0: SABR alpha at t = 0
    :param beta: SABR beta
    :param nu: SABR rho
    :param rho: SABR nu
    :return: the Black implied volatility, computed using the Hagan et al. lognormal appoximation
    '''

    one_beta = 1.0 - beta
    one_betasqr = one_beta * one_beta
    if F_0 != S:
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
