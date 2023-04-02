module Analysis

export portfolio_risk
export portfolio_return
export portfolio_log_return
export portfolio_performance

    using LinearAlgebra
    using Statistics
    using Distributions

    # Function to calculate the risk of a portfolio
    function portfolio_risk(returns, weights, risk_measure; alpha=nothing, target=nothing)
        """
        Calculate the risk of a portfolio given the historical returns, weights, and a risk measure function.
        """
        portfolio_returns = returns * weights
        if alpha !== nothing
            risk = risk_measure(portfolio_returns, alpha)
        elseif target !== nothing
            risk = risk_measure(portfolio_returns, target)
        else
            risk = risk_measure(portfolio_returns)
        end
        return risk
    end

    # Function to calculate the return of a portfolio
    function portfolio_return(returns, weights)
        """
        Calculate the return of a portfolio given the historical returns and weights.
        """
        asset_returns = mean.(eachcol(returns))
        port_return = dot(weights, asset_returns)
        return port_return
    end

    # Function to calculate the logarithmic return of a portfolio
    function portfolio_log_return(returns, weights)
        """
        Calculate the logarithmic return of a portfolio given the historical returns and weights.
        """
        log_returns = log.(returns)
        weighted_log_returns = log_returns * weights
        port_log_return = mean(weighted_log_returns)
        return port_log_return
    end

    # Function to analyze the overall performance of a portfolio
    function portfolio_performance(returns, weights, risk_measure; log_return=false, alpha=nothing, target=nothing)
        """
        Analyze the performance of a portfolio given the historical returns, weights, and a risk measure function.
        Returns the portfolio risk and return.
        """
        port_risk = portfolio_risk(returns, weights, risk_measure, alpha=alpha, target=target)
        if log_return
            port_return = portfolio_log_return(returns, weights)
        else
            port_return = portfolio_return(returns, weights)
        end
        return port_risk, port_return
    end

end
