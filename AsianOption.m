% Simulating the Stock Price Paths

function SPath = BS_SP_Path(S0, params, T, NSteps, NPaths)
    mu = params(1); sigma = params(2);

    dt = T/NSteps;
    vet = (0:dt:T);
    mut = (mu - 0.5*sigma^2)*vet;

    BMV = cumsum(sqrt(dt)*randn(NSteps, NPaths));
    BMV = [zeros(1, NPaths); BMV];

    SPath = S0*exp(mut * ones(1, NPaths) + sigma * BMV);
end


% Function: pricing asian option with control variate is based on 
% the payoff of the mathematic average Asian option

function [P, CI] = AsianMCCV1(S0, K, T, params, NSamples, NRep, NPilot)
    % pilot replications to set control parameter r = params(1);
    %sigma = params(2);
    TryPath = BS_SP_Path(S0, params, T, NSamples, NPilot); 
    StockSum = sum(TryPath, 1) ;
    TryPayoff = mean(TryPath(2:end, :), 1) ;
    TryPayoff = exp(-r*T) * max(0, TryPayoff - K);
    MatCov = cov(StockSum, TryPayoff); 
    cstar = -MatCov(1, 2) / var(StockSum); 
    dt = T / NSamples;
    ExpSum = S0 * (1 - exp((NSamples + 1)*r*dt)) / (1 - exp(r*dt));
    
    % Monte Carlo runs 
    ControlVars = zeros(NRep, 1);
    for i = 1:NRep
        StockPath = BS_SP_Path(S0, params, T, NSamples, 1); 
        Payoff = exp(-r*T) * max(0, mean(StockPath(2:end)) - K); 
        ControlVars(i) = Payoff + cstar * (sum(StockPath) - ExpSum);
    end
[P, aux, CI] = normfit(ControlVars);
end


% Function: pricing asian option with control variate is based on 
% the payoff of the geometric average Asian option

function [P, CI] = AsianMCGeoCV(S0, K, T, params, NSamples, NRep, NPilot)
    % precompute quantities r = params(1);
    DF = exp(-r*T);
    GeoExact = GeometricAsian(S0, K, T, params, NSamples); 
    GeoPrices = zeros(NPilot, 1);
    AriPrices = zeros(NPilot, 1);

    for i = 1:NPilot
        Path = BS_SP_Path(S0, params, T, NSamples, 1);
        GeoPrices(i) = DF*max(0, (prod (Path(2:end) ) )^(1/NSamples) - K); 
        AriPrices(i) = DF*max(0, mean(Path(2:end)) - K);
    end
    MatCov = cov(GeoPrices, AriPrices); 
    cstar = -MatCov(1,2) / var(GeoPrices);
    
    % Monte Carlo runs 
    ControlVars = zeros(NRep, 1); 
    for i = 1:NRep
        Path = BS_SP_Path(S0, params, T, NSamples, 1) ;
        GeoPrice = DF*max(0, (prod(Path(2:end)))^(1/NSamples) - K); 
        AriPrice = DF*max(0, mean(Path(2:end)) - K);
        ControlVars(i) = AriPrice + cstar * (GeoPrice - GeoExact);
    end
 
[P, aux, CI] = normfit(ControlVars);
end



