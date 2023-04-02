module Measures

export standard_deviation
export sqrt_kurtosis
export mean_absolute_deviation
export gini_mean_difference
export cvar_range
export tail_gini_range
export range_of_returns
export semi_std_deviation
export sqrt_semi_kurtosis
export first_lower_partial_moment
export second_lower_partial_moment
export value_at_risk
export cvar
export tail_gini
export worst_case_realization
export maximum_drawdown
export calmar_ratio
export average_drawdown
export cdrawdown_at_risk
export ulcer_index

    using Statistics
    using LinearAlgebra
    using Distributions
    using Roots

    function standard_deviation(returns)
    """
    Calculate the standard deviation of the given `returns` vector.
    Standard deviation is a measure of the dispersion of returns and is commonly used as a proxy for risk.
    """
        return std(returns)
    end

    function sqrt_kurtosis(returns)
    """
    Calculate the square root kurtosis of the given `returns` vector.
    Square root kurtosis is a measure of the tail risk, indicating the likelihood of extreme events.
    """
        return sqrt(abs(kurtosis(returns)))
    end

    function mean_absolute_deviation(returns)
    """
    Calculate the mean absolute deviation of the given `returns` vector.
    Mean absolute deviation is another measure of dispersion, indicating the average deviation of returns from their mean.
    """
        mean_return = mean(returns)
        return mean(abs.(returns .- mean_return))
    end

    function gini_mean_difference(returns)
    """
    Calculate the Gini Mean Difference (GMD) of the given `returns` vector.
    Gini Mean Difference measures the average absolute difference between all pairs of returns, providing a measure of the dispersion of returns.
    """
        n = length(returns)
        return sum([abs(returns[i] - returns[j]) for i in 1:n, j in 1:n]) / (n^2)
    end

    function cvar_range(returns, alpha=0.05)
    """
    Calculate the Conditional Value at Risk (CVaR) Range of the given `returns` vector at the specified `alpha` level.
    CVaR Range is the difference between the CVaR values at the specified `alpha` level and its complement (1-alpha). It measures the dispersion of extreme losses.
    """
        VaR = quantile(returns, 1-alpha)
        return mean(returns[returns .<= VaR])
    end

    function tail_gini_range(returns, alpha=0.05)
    """
    Calculate the Tail Gini Range of the given `returns` vector at the specified `alpha` level.
    Tail Gini Range measures the difference in tail inequality between the lower and upper tails at the specified `alpha` level, providing a measure of the asymmetry of extreme returns.
    """
        VaR = quantile(returns, 1-alpha)
        tail_returns = returns[returns .<= VaR]
        return gini_mean_difference(tail_returns)
    end

    function range_of_returns(returns)
    """
    Calculate the range of the given `returns` vector.
    Range is the difference between the maximum and minimum return values, representing the overall dispersion of returns.
    """
        return maximum(returns) - minimum(returns)
    end

    function semi_std_deviation(returns)
    """
    Calculate the semi-standard deviation of the given `returns` vector.
    Semi-standard deviation measures the dispersion of returns that are below the mean, focusing on downside risk.
    """
        downside_returns = returns[returns .< mean(returns)]
        return std(downside_returns)
    end

    function sqrt_semi_kurtosis(returns)
    """
    Calculate the square root of semi kurtosis of the given `returns` vector.
    Square root semi kurtosis is a measure of the tail risk, indicating the likelihood of extreme events in the downside tail.
    """
        downside_returns = returns[returns .< 0]
        return sqrt_kurtosis(downside_returns)
    end

    function first_lower_partial_moment(returns, target=0)
    """
    Calculate the first lower partial moment of the given `returns` vector with respect to the `threshold`.
    First lower partial moment measures the average return below the threshold and is used in the calculation of the Omega Ratio.
    """
        downside_returns = returns[returns .< target]
        return mean(downside_returns .- target)
    end

    function second_lower_partial_moment(returns, target=0)
    """
    Calculate the second lower partial moment of the given `returns` vector with respect to the `threshold`.
    Second lower partial moment measures the average squared return below the threshold and is used in the calculation of the Sortino Ratio.
    """
        downside_returns = returns[returns .< target]
        return mean((downside_returns .- target).^2)
    end

    function value_at_risk(returns, alpha=0.05)
    """
    Calculate the Value at Risk (VaR) of the given `returns` vector at the specified `alpha` level.
    VaR measures the potential loss in value of a portfolio over a defined period for a given confidence interval.
    """
        return quantile(returns, alpha)
    end

    function cvar(returns, alpha=0.05)
    """
    Calculate the Conditional Value at Risk (CVaR) of the given `returns` vector at the specified `alpha` level.
    CVaR measures the expected value of the losses beyond the Value at Risk (VaR) threshold, focusing on extreme losses.
    """
        return cvar_range(returns, alpha)
    end

    function tail_gini(returns, alpha=0.05)
    """
    Calculate the Tail Gini of the given `returns` vector at the specified `alpha` level.
    Tail Gini measures the inequality in the tail of the return distribution, providing a measure of the concentration of extreme returns.
    """
        return tail_gini_range(returns, alpha)
    end

    function worst_case_realization(returns)
    """
    Calculate the worst case realization (minimax) of the given `returns` vector.
    Worst case realization is the minimum return observed in the dataset, representing the most extreme negative return.
    """
        return minimum(returns)
    end

    function maximum_drawdown(returns)
    """
    Calculate the maximum drawdown for the given `returns` vector based on uncompounded cumulative returns.
    Maximum drawdown measures the largest peak-to-trough decline in the cumulative returns, indicating the worst possible loss over a given time period.
    """
        cumulative_returns = cumsum(returns)
        return maximum([cumulative_returns[i] - maximum(cumulative_returns[1:i]) for i in 1:length(cumulative_returns)]) / maximum(cumulative_returns)
    end

    function calmar_ratio(returns)
    """
    Calculate the Calmar Ratio for the given `returns` vector based on uncompounded cumulative returns.
    The Calmar Ratio is a risk-adjusted performance measure that compares the average return to the maximum drawdown.
    """
        mean_return = mean(returns)
        max_drawdown = maximum_drawdown(returns)
        return mean_return / max_drawdown
    end

    function average_drawdown(returns)
    """
    Calculate the average drawdown for the given `returns` vector based on uncompounded cumulative returns.
    Average drawdown measures the mean of all drawdowns, providing an indication of the typical drawdown experienced over the given time period.
    """
        cumulative_returns = cumsum(returns)
        drawdown_sum = 0.0
        peak_value = cumulative_returns[1]
        drawdown_count = 0

        for i in 2:length(cumulative_returns)
            if cumulative_returns[i] < peak_value
                drawdown = (peak_value - cumulative_returns[i]) / peak_value
                drawdown_sum += drawdown
                drawdown_count += 1
            else
                peak_value = cumulative_returns[i]
            end
        end

        return drawdown_sum / drawdown_count
    end

    function cdrawdown_at_risk(returns, alpha=0.05)
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of the given `returns` vector at the specified `alpha` level.
    CDaR measures the average drawdown beyond a certain drawdown threshold, focusing on extreme drawdowns.
    """
        cumulative_returns = cumsum(returns) # Calculate cumulative returns
        drawdowns = [maximum(cumulative_returns[1:i]) - cumulative_returns[i] for i in 1:length(cumulative_returns)] # Change from returns to cumulative_returns
        return quantile(drawdowns, alpha)
    end

    function ulcer_index(returns)
    """
    Calculate the Ulcer Index of the given `returns` vector.
    Ulcer Index measures the depth and duration of drawdowns, providing a measure of risk-adjusted performance.
    """
        cumulative_returns = cumsum(returns) # Calculate cumulative returns
        drawdowns = [maximum(cumulative_returns[1:i]) - cumulative_returns[i] for i in 1:length(cumulative_returns)] # Change from returns to cumulative_returns
        return sqrt(mean(drawdowns.^2))
    end 
 
end