GitHub main.yml Status | Travis CI building Status | Stable Documentation | Dev Documentation
----------|-----------|----------|-----------
![.github/workflows/main.yml](https://github.com/MohHizzani/NumNN.jl/workflows/.github/workflows/main.yml/badge.svg) | [![Build Status](https://travis-ci.com/MohHizzani/NumNN.jl.svg?branch=master)](https://travis-ci.com/MohHizzani/NumNN.jl) | [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mohhizzani.github.io/NumNN.jl/stable) | [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://mohhizzani.github.io/NumNN.jl/dev)

# NumNN.jl

This package provides high-level Neural Network APIs deals with different number representations like [Posit][1], Logarithmic Data Representations, Residual Number System (RNS), and -for sure- the conventional IEEE formats.

Since, the implementation and development process for testing novel number systems on different Deep Learning applications using the current available DP frameworks in easily feasible. An urgent need for an unconventional library that provides both the easiness and complexity of simulating and testing and evaluate new number systems before the hardware design complex process— was resurfaced.



## Why Julia?

[Julia][2] provides in an unconventional way the ability to simulate new number systems and deploy this simulation to be used as high-level primitive type. **[Multiple Dispatch][3]** provides a unique ability to write a general code then specify the implementation based on the type.

### Examples of Multiple Dispatch

```julia
julia> aInt = 1; #with Integer type

julia> bInt = 2; #with Integer type

julia> cInt = 1 + 2; #which is a shortcut for cInt = +(1,2)

julia> aFloat = 1.0; #with Float64 type

julia> bFloat = 2.0; #with Flot64 type

julia> aInt + bFloat #will use the method +(::Int64, ::Float64)
3.0
```

Now let's do something more interesting with **Posit** (continue on the previous example)

```julia
julia> using SoftPosit

julia> aP = Posit16(1)
Posit16(0x4000)

julia> bP = Posit16(2)
Posit16(0x5000)

julia> aInt + aP #Note how the result is in Posit16 type
Posit16(0x5000)
```

The output was of type `Posit16` because in **Julia** you can define a [Promote Rule][4] which mean when the output can be in either type(s) of the input, **promote** the output to be of the specified type. Which can be defined as follows:

```julia
Base.promote_rule(::Type{Int64}, ::Type{Posit16}) = Posit16
```

This means that the output of an operation on both `Int64` and `Posit16` should be converted to `Posit16`.


## Install

To install in Julia

```julia
julia> ] add NumNN
```
## To Use

```julia
julia> using NumNN
```


[1]: <superfri.org/superfri/article/view/137> "Beating Floating Point at its Own Game: Posit Arithmetic"
[2]: <julialang.org> "Julia Language"
[3]: <https://docs.julialang.org/en/v1/manual/methods/> "Julia Multiple Dispatch"
[4]: <https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#Promotion-1> "Juila Promotion"
