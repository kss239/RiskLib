module Optimizers

export mean_risk_optimization
export logarithmic_mean_risk_optimization
export risk_parity_optimization

    using LinearAlgebra
    using Statistics
    using JuMP
    using Ipopt

    function mean_risk_optimization(returns, risk_measure, obj_type; lambda=nothing, alpha=nothing, target=nothing)
        """
        Optimize a portfolio using a Mean-Risk optimization approach.
        The risk measure should be a convex function from `measures.jl`.
        `lambda` is the risk aversion parameter.
        `obj_type` should be :minimize, :maximize, :utility, or :ratio
        """
        @assert obj_type in [:minimize, :maximize, :utility, :ratio] "obj_type should be :minimize, :maximize, :utility, or :ratio"
        n = size(returns, 2)

        risk_func(w, returns) = if alpha !== nothing
            risk_measure(returns * w, alpha)
        elseif target !== nothing
            risk_measure(returns * w, target)
        else
            risk_measure(returns * w)
        end

        model = Model(Ipopt.Optimizer)
        @variable(model, w[1:n] >= 0)
        @constraint(model, sum(w) == 1)

        if obj_type == :minimize
                @objective(model, Min, risk_func(w, returns))
        elseif obj_type == :maximize
                @objective(model, Max, sum(w[i] * mean(returns[:, i]) for i in 1:n))
        elseif obj_type == :utility
            @assert lambda !== nothing "Please provide a value for `lambda` when using :utility objective"
            @objective(model, Max, sum(w[i] * mean(returns[:, i]) for i in 1:n) - lambda * risk_func(w, returns))
        elseif obj_type == :ratio
            @objective(model, Max, (sum(w[i] * mean(returns[:, i]) for i in 1:n)) / risk_func(w, returns))
        end

        optimize!(model)

        opt_w = value.(w)
        return opt_w
    end

    function logarithmic_mean_risk_optimization(returns, risk_measure, obj_type; lambda=nothing, alpha=nothing, target=nothing)
        """
        Optimize a portfolio using a Logarithmic Mean-Risk optimization approach.
        The risk measure should be a convex function from `measures.jl`.
        `lambda` is the risk aversion parameter.
        `obj_type` should be :minimize, :maximize, :utility, or :ratio
        """
        @assert obj_type in [:minimize, :maximize, :utility, :ratio] "obj_type should be :minimize, :maximize, :utility, or :ratio"
        n = size(returns, 2)

        risk_func(w, returns) = if alpha !== nothing
            risk_measure(returns * w, alpha)
        elseif target !== nothing
            risk_measure(returns * w, target)
        else
            risk_measure(returns * w)
        end

        model = Model(Ipopt.Optimizer)
        @variable(model, w[1:n] >= 0)
        @constraint(model, sum(w) == 1)

        if obj_type == :minimize
                @objective(model, Min, risk_func(w, returns))
        elseif obj_type == :maximize
                @objective(model, Max, sum(w[i] * log(mean(returns[:, i])) for i in 1:n))
        elseif obj_type == :utility
            @assert lambda !== nothing "Please provide a value for `lambda` when using :utility objective"
            @objective(model, Max, sum(w[i] * log(mean(returns[:, i])) for i in 1:n) - lambda * risk_func(w, returns))
        elseif obj_type == :ratio
            @objective(model, Max, sum(w[i] * log(mean(returns[:, i])) for i in 1:n) / risk_func(w, returns))
        end


        optimize!(model)

        opt_w = value.(w)
        return opt_w
    end

    function kelly_criterion_optimization(returns)
        """
        Optimize a portfolio using the Kelly Criterion.
        """
        n = size(returns, 2)

        model = Model(Ipopt.Optimizer)
        @variable(model, w[1:n] >= 0)
        @constraint(model, sum(w) == 1)

        # Objective function: maximize the expected logarithmic return
        @objective(model, Max, sum(w[i] * log(mean(returns[:, i])) for i in 1:n))

        optimize!(model)

        opt_w = value.(w)
        return opt_w
    end


    function risk_parity_optimization(returns, risk_measure, obj_type; lambda=nothing, alpha=nothing, target=nothing)
        """
        Optimize a portfolio using the Risk Parity approach.
        The risk measure should be a convex function from `measures.jl`.
        `lambda` is the risk aversion parameter.
        `obj_type` should be :minimize, :maximize, :utility, or :ratio
        """
        @assert obj_type in [:minimize, :maximize, :utility, :ratio] "obj_type should be :minimize, :maximize, :utility, or :ratio"

        n = size(returns, 2)

        risk_func(w, returns) = if alpha !== nothing
            risk_measure(returns * w, alpha)
        elseif target !== nothing
            risk_measure(returns * w, target)
        else
            risk_measure(returns * w)
        end

        model = Model(Ipopt.Optimizer)
        @variable(model, w[1:n] >= 0)
        @constraint(model, sum(w) == 1)

        # Calculate the risk measure for each asset
        risks = [risk_func(reshape(returns[:, i], :, 1), returns) for i in 1:n]

        # Objective function
        if obj_type == :minimize
            @objective(model, Min, sum((w[i] * risks[i] - target)^2 for i in 1:n))
        elseif obj_type == :maximize
            @objective(model, Max, sum(w[i] * mean(returns[:, i]) for i in 1:n))
        elseif obj_type == :utility
            @assert lambda !== nothing "Please provide a value for `lambda` when using :utility objective"
            @objective(model, Max, sum(w[i] * mean(returns[:, i]) for i in 1:n) - lambda * sum((w[i] * risks[i] - target)^2 for i in 1:n))
        elseif obj_type == :ratio
            @objective(model, Max, sum(w[i] * mean(returns[:, i]) for i in 1:n) / sum((w[i] * risks[i] - target)^2 for i in 1:n))
        end

        optimize!(model)

        opt_w = value.(w)
        return opt_w
    end

                                                                                                        
end
