module Optimizers

export mean_risk_optimization_montecarlo
export risk_parity_optimization_montecarlo

    using LinearAlgebra
    using Statistics

    using Random
    using Distributions                                                                                             

    function mean_risk_optimization_montecarlo(returns, risk_measure, obj_type; lambda=nothing, alpha=nothing, target=nothing, n_samples=10000)
        """
        Optimize a portfolio using a Mean-Risk optimization approach with Monte Carlo simulation.
        The risk measure should be a convex function from `measures.jl`.
        `lambda` is the risk aversion parameter.
        `obj_type` should be :minimize, :maximize, :utility, or :ratio
        """
        @assert obj_type in [:minimize, :maximize, :utility, :ratio] "obj_type should be :minimize, :maximize, :utility, or :ratio"
        n = size(returns, 2)

        risk_func(w, returns; alpha=nothing, target=nothing) = if alpha !== nothing
            risk_measure(returns * w, alpha)
        elseif target !== nothing
            risk_measure(returns * w, target)
        else
            risk_measure(returns * w)
        end

        function objective(w, returns, obj_type)
            if obj_type == :minimize
                return risk_func(w, returns, alpha=alpha, target=target)
            elseif obj_type == :maximize
                return sum(w[i] * mean(returns[:, i]) for i in 1:n)
            elseif obj_type == :utility
                return sum(w[i] * mean(returns[:, i]) for i in 1:n) - lambda * risk_func(w, returns)
            elseif obj_type == :ratio
                risk = risk_func(w, returns)
                expected_return = sum(w[i] * mean(returns[:, i]) for i in 1:n)
                return -expected_return / risk
            end
        end

        best_w = nothing
        best_obj_value = obj_type == :minimize ? Inf : -Inf

        for _ in 1:n_samples
            w = rand(Dirichlet(ones(n)))
            obj_value = objective(w, returns, obj_type)

            if (obj_type == :minimize && obj_value < best_obj_value) || (obj_type != :minimize && obj_value > best_obj_value)
                best_w = w
                best_obj_value = obj_value
            end
        end

        return best_w
    end

    function risk_parity_optimization_montecarlo(returns, risk_measure, obj_type, risk_lvl=0; lambda=nothing, alpha=nothing, target=nothing, n_samples=10000)
        """
        Optimize a portfolio using the Risk Parity approach and Monte Carlo optimization.
        The risk measure should be a convex function from `measures.jl`.
        `lambda` is the risk aversion parameter.
        `obj_type` should be :minimize, :maximize, :utility, or :ratio
        """
        @assert obj_type in [:minimize, :maximize, :utility, :ratio] "obj_type should be :minimize, :maximize, :utility, or :ratio"

        n = size(returns, 2)

        risk_func(w, returns; alpha=nothing, target=nothing) = if alpha !== nothing
            risk_measure(returns * w, alpha)
        elseif target !== nothing
            risk_measure(returns * w, target)
        else
            risk_measure(returns * w)
        end

        # Generate random portfolio weights
        function generate_random_weights(n)
            w = rand(n)
            return w / sum(w)
        end

        best_value = -Inf
        best_weights = zeros(n)

        for _ in 1:n_samples
            w = generate_random_weights(n)

            # Calculate the objective function value based on the specified objective type
            if obj_type == :minimize
                value = sum((w[i] * risk_func(w, returns, alpha = alpha, target = target) - risk_lvl)^2 for i in 1:n)
            elseif obj_type == :maximize
                value = sum(w[i] * mean(returns[:, i]) for i in 1:n)
            elseif obj_type == :utility
                @assert lambda !== nothing "Please provide a value for `lambda` when using :utility objective"
                value = sum(w[i] * mean(returns[:, i]) for i in 1:n) - lambda * sum((w[i] * risk_func(w, returns, alpha = alpha, target = target) - risk_lvl)^2 for i in 1:n)
            elseif obj_type == :ratio
                risk = sum((w[i] * risk_func(w, returns, alpha = alpha, target = target) - risk_lvl)^2 for i in 1:n)
                expected_return = sum(w[i] * mean(returns[:, i]) for i in 1:n)
                value = expected_return / risk
            end

            # Update the best value and weights if necessary
            if obj_type in [:maximize, :utility, :ratio] && value > best_value
                best_value = value
                best_weights = w
            elseif obj_type == :minimize && value < best_value
                best_value = value
                best_weights = w
            end
        end

        return best_weights
    end
                                                                                                        
end