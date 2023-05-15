
module Analysis

export portfolio_risk
export portfolio_return
export portfolio_log_return
export portfolio_performance
export efficient_frontier
export monte_carlo_scenario_analysis

    using LinearAlgebra
    using Statistics
    using Plots
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

    function efficient_frontier(returns, risk_measure, target_returns, optimization_function; lambda=1.0, alpha=nothing, plot_frontier=true)
        """
        Generate the efficient frontier for a given set of assets, returns, and risk measure using the specified optimization function.
        `returns` is a matrix of asset returns.
        `risk_measure` is a risk metric function from `measures.jl`.
        `target_returns` is a vector of target returns for which to compute the optimal portfolios.
        `optimization_function` is an optimization function from the `Optimizers` module.
        `lambda` is the risk aversion parameter.
        `alpha` is the parameter for the risk measure (if applicable).
        `plot_frontier` is a boolean indicating whether to plot the efficient frontier.
        """
        n = size(returns, 2)
        frontier_risk = []
        optimal_weights = []

        for target in target_returns
            opt_w = optimization_function(returns, risk_measure, :utility, lambda=lambda, alpha=alpha, target=target)
            push!(optimal_weights, opt_w)
            port_risk = risk_measure(returns * opt_w, alpha)
            push!(frontier_risk, port_risk)
        end

        if plot_frontier
            p = plot(target_returns, frontier_risk, xlabel="Return", ylabel="Risk", title="Efficient Frontier", legend=false)
            display(p)
        end

        return optimal_weights, target_returns, frontier_risk
    end

    function monte_carlo_scenario_analysis(returns, portfolio_weights, risk_measure, num_simulations, num_periods; alpha=nothing)
        """
        Perform Monte Carlo scenario analysis on a portfolio.

        Args:
        - returns: The historical returns data for all assets in the portfolio.
        - portfolio_weights: The weights of the assets in the portfolio.
        - risk_measure: The risk measure function from `measures.jl`.
        - num_simulations: The number of Monte Carlo simulations to run.
        - num_periods: The number of periods to simulate in each scenario.
        - alpha (optional): The alpha value for the risk measure, if applicable.

        Returns:
        - A list of risk values for each simulation.
        """
        n = size(returns, 2)
        cov_matrix = cov(returns)
        mean_returns = vec(mean(returns, dims=1))

        risk_values = zeros(num_simulations)

        for i in 1:num_simulations
            # Generate random returns for each asset based on their historical mean and covariance
            simulated_returns = rand(MvNormal(mean_returns, cov_matrix), num_periods)

            # Calculate the portfolio returns for the simulated scenario
            portfolio_simulated_returns = simulated_returns * portfolio_weights

            # Compute the risk value for the simulated scenario
            if alpha !== nothing
                risk_value = risk_measure(portfolio_simulated_returns, alpha)
            else
                risk_value = risk_measure(portfolio_simulated_returns)
            end

            risk_values[i] = risk_value
        end

        return risk_values
    end

end
