module tdse

using LinearAlgebra
using Quadmath

mutable struct dgatom
    coef::Float64
    σ::Float64
    μ::Float64
end

mutable struct dwavefun
    nterms::Int
    atoms::Array{dgatom,1}
end

function dgatom_val(g::dgatom,x::Array{Float64,1})::Array{Float64,1}
    return g.coef./(pi*g.σ)^0.25.*exp.(-(x.-g.μ).^2.0./g.σ./2.0)
end

function dwavefun_val(w::dwavefun,x::Array{Float64,1})::Array{Float64,1}
    return sum(dgatom_val(w.atoms[i],x) for i = 1:w.nterms)
end

function dgatom_secder(g::dgatom,x::Array{Float64,1},)::Array{Float64,1}
    return ((x.-g.μ).^2.0./g.σ./g.σ.-1.0./g.σ).*dgatom_val(g,x)
end

function dinnerprod(g1::dgatom,g2::dgatom;use_coef=false)::Float64
    if use_coef
        return g1.coef*g2.coef*sqrt(2.0)*(g1.σ*g2.σ)^0.25/
            sqrt(g1.σ+g2.σ)*exp(-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)/2.0)
    else
        return sqrt(2.0)*(g1.σ*g2.σ)^0.25/sqrt(g1.σ+g2.σ)*
            exp(-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)/2.0)
    end
end


function dconvgatoms(k::dgatom,g::dgatom)::dgatom
    """
    Compute the convolution of two Gaussian atoms
    int_{R}k(x-y)g(y)dy
    where the kernel k(x-y) = coef*gatom(x,σ,y)
    and g(x) = coef*gatom(y,σ,μ).
    """
    coef = (4.0*pi*k.σ*g.σ/(k.σ+g.σ))^0.25*k.coef*g.coef
    σ = k.σ+g.σ
    μ = g.μ
    return dgatom(coef,σ,μ)
end

function dprodgatoms(g1::dgatom,g2::dgatom)::dgatom
    coef = g1.coef*g2.coef/(pi*(g1.σ+g2.σ))^0.25*
            exp(-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)/2.0)
    σ = g1.σ*g2.σ/(g1.σ+g2.σ)
    μ = (g1.σ*g2.μ+g2.σ*g1.μ)/(g1.σ+g2.σ)
    return dgatom(coef,σ,μ)
end

function dgradinnerprod(g1::dgatom,g2::dgatom;use_coef=false)::Float64
    """
    Compute the inner product of the gradients
    int_{R}  d/dx(f(x)) * d/dx(g(x)) dx
    """
    if use_coef
        return g1.coef*g2.coef*(4.0*g1.σ*g2.σ)^0.25/sqrt(g1.σ+g2.σ)*
            (-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)^2.0+1.0/(g1.σ+g2.σ))*
            exp(-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)/2.0)
    else
        return (4.0*g1.σ*g2.σ)^0.25/sqrt(g1.σ+g2.σ)*
            (-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)^2.0+1.0/(g1.σ+g2.σ))*
            exp(-(g1.μ-g2.μ)^2.0/(g1.σ+g2.σ)/2.0)
    end
end

function doptexp(acc,dnear,dfar)
    """
    Generates an approximation
    f(r) = sum_{j=1}^{nt} wei_j exp(- pp_j r^2)
    with the number of terms "nt", so that
    |f(r) - exp(-r)| <= acc for dnear <= r <= dfar.
    """
    eps = 1e-17
    aa = 0.037037037037037037
    bb = -0.253086419753086420

    # set stepsize
    hstep = 1.0/(aa+bb*log(acc)/log(10.0))

    # generate an equally spaced logarithmic grid
    rstart = -73.0
    rstop = 5.0
    nstep = ceil(Int,(rstop-rstart)/hstep)
    nterms = nstep

    # generate an excessive collection of gaussians
    ppini = zeros(Float64,nterms)
    weiini = zeros(Float64,nterms)
    for i = 0:nterms-1
        ppini[nterms-i] = exp(-rstart-i*hstep)
        weiini[nterms-i] = hstep/2.0/sqrt(pi)*
            exp(-exp(rstart+i*hstep)/4.0+(rstart+i*hstep)/2.0)
    end

    # evaluate error of approximation at lower limit "dnear" and drop terms
    ss = 0.0
    nstop = 0
    for i = 1:nterms
        ss += weiini[i]*exp(-ppini[i]*dnear*dnear)
        err = log(abs(ss*exp(dnear)-1.0)+eps)/log(10.0)

        if err < log(acc)/log(10)
            break
        else
            nstop += 1
        end
    end

    nt = nstop
    pp = ppini[1:nt]
    wei = weiini[1:nt]

    return nt,pp,wei

end


function sfastchol(ϕ::dwavefun,krankmax::Int,eps::Float64)
    """
    Reduce the number of terms in a wave function that is
    represented as a Gaussian mixture.
    The reduced mixture has up to 7~8 digits of accuracy
    (depending on the input eps).

    Inputs:

    ϕ             --- wave function to be reduced
    krankmax      --- maxiμm expected number of terms;
                      if exceeded info = 1 on exit, otherwise info = 0

    eps           --- accuracy, the resulting accuracy is eps,
                      eps^2 can be 1d-14 - 1d-15 or so
                      if ifl = 1 then it is reset to the value reached
                      at krankmax

    Outputs:

    ipivot        --- pivot vector
    ϕ             --- reduced representation

    info          --- info = 1 if number of terms exceeds krankmax,
                      info = 0, otherwise
    """

    # set error threshold
    eps2 = eps*eps
    if eps2 < 1e-15
        eps2 = 1e-15
        @warn "input eps < 1e-7, eps2 is reset to 1e-15"
    end

    L = Array{Float64,2}(undef,krankmax,ϕ.nterms)
    diag = ones(Float64,ϕ.nterms)
    ipivot = Int[1:ϕ.nterms;]
    newnterms = 0

    for i = 1:ϕ.nterms
        # find largest diagonal element
        dmax = diag[ipivot[i]]
        imax = i
        for j = i+1:ϕ.nterms
            if dmax < diag[ipivot[j]]
                dmax = diag[ipivot[j]]
                imax = j
            end
        end

        # swap to the leading position
        ipivot[i],ipivot[imax] = ipivot[imax],ipivot[i]

        # check if the diagonal element large enough
        if diag[ipivot[i]] < eps2
            info = 0
            break
        end

        L[i,ipivot[i]] = sqrt(diag[ipivot[i]])

        for j = i+1:ϕ.nterms
            r1 = dot(L[1:i-1, ipivot[i]], L[1:i-1, ipivot[j]])
            r2 = dinnerprod(ϕ.atoms[ipivot[i]],ϕ.atoms[ipivot[j]])
            L[i,ipivot[j]] = (r2-r1)/L[i,ipivot[i]]
            diag[ipivot[j]] -= L[i,ipivot[j]]^2.0
        end

        newnterms += 1
    end

    # solve for new coefficients using forward/backward substitution
    newcoefs = zeros(Float64,newnterms)

    for i = 1:newnterms
        for j = newnterms+1:ϕ.nterms
            newcoefs[i] += ϕ.atoms[ipivot[j]].coef*
                            dot(L[1:i,ipivot[i]],L[1:i,ipivot[j]])
        end
    end

    # forward substitution
    for i = 1:newnterms
        r1 = 0.
        for j = 1:i-1
            r1 += L[j,ipivot[i]]*newcoefs[j]
        end
        newcoefs[i] = (newcoefs[i]-r1)/L[i,ipivot[i]]
    end

    # backward substitution
    for i = newnterms:-1:1
        r1 = 0.
        for j = i+1:newnterms
            r1 += L[i,ipivot[j]]*newcoefs[j]
        end
        newcoefs[i] = (newcoefs[i]-r1)/L[i,ipivot[i]]
    end

    # add "skeleton" terms and copy exponents and shifts
    newϕ = dwavefun(newnterms,Array{dgatom,1}(undef,newnterms))
    for i = 1:newnterms
        newϕ.atoms[i] = dgatom(newcoefs[i]+ϕ.atoms[ipivot[i]].coef,
                                ϕ.atoms[ipivot[i]].σ,
                                ϕ.atoms[ipivot[i]].μ)
    end

    return newϕ
end


function tise_solve(;maxiter=20)
    """
    Solve time-independent Schrodinger equation
    """

    # get representation of exp(-r) as sum of Gaussians
    nt,pp,wei = doptexp(1e-8,1e-8,1e3)

    # get potential v(x) = -8e^(-x^2)
    v = dwavefun(1,Array{dgatom,1}(undef,1))
    v.atoms[1] = dgatom(-8.0*(pi/2.)^0.25,0.5,0.)

    # get initial solution
    # ϕ0 = wavefun(1,Array{gatom,1}(undef,1))
    # ϕ0.atoms[1] = gatom(1.,1.,0.)

    ϕ0 = dwavefun(2,Array{dgatom,1}(undef,2))
    ϕ0.atoms[1] = dgatom(1.,1.,-1.)
    ϕ0.atoms[2] = dgatom(1.,1.,1.)

    # compute energy
    energy = .5*sum(dgradinnerprod(ϕ0.atoms[i],ϕ0.atoms[j],use_coef=true)
            for i=1:ϕ0.nterms,j=1:ϕ0.nterms)+
            sum(dinnerprod(dprodgatoms(v.atoms[i],ϕ0.atoms[j]),ϕ0.atoms[k],use_coef=true)
            for i=1:v.nterms,j=1:ϕ0.nterms,k=1:ϕ0.nterms)

    μ = sqrt(-2.0*energy)
    println(energy,"   ",μ)

    eps = 1e-10
    for iter = 1:maxiter

        # compute vϕ = v*ϕ0
        ntermsmax = v.nterms*ϕ0.nterms
        vϕ = dwavefun(ntermsmax,Array{dgatom,1}(undef,ntermsmax))

        icount = 0
        for i = 1:v.nterms
            for j = 1:ϕ0.nterms
                vϕ.atoms[icount+1] = dprodgatoms(v.atoms[i],ϕ0.atoms[j])
                if abs(vϕ.atoms[icount+1].coef) > eps
                    icount += 1
                end
            end
        end
        vϕ.nterms = icount

        # reduce number of terms in vϕ
        krankmax = 1000
        vϕ = sfastchol(vϕ,krankmax,1e-7)

        # get Green's function
        gf = dwavefun(nt,Array{dgatom,1}(undef,nt))
        for i = 1:nt
            gf.atoms[i] =
                dgatom(-1.0/μ*wei[i]*(pi/(2.0*μ*μ*pp[i]))^.25,.5/μ/μ/pp[i],0.)
        end

        ntermsmax = gf.nterms*vϕ.nterms
        ϕ1 = dwavefun(ntermsmax,Array{dgatom,1}(undef,ntermsmax))
        icount = 0
        for i = 1:gf.nterms
            for j = 1:vϕ.nterms
                ϕ1.atoms[icount+1] = dconvgatoms(gf.atoms[i],vϕ.atoms[j])
                if abs(ϕ1.atoms[icount+1].coef) > eps
                    icount += 1
                end
            end
        end
        ϕ1.nterms = icount

        krankmax = 1000
        ϕ1 = sfastchol(ϕ1,krankmax,1e-7)

        # normalize ϕ1
        ϕ1_norm = sum(dinnerprod(ϕ1.atoms[i],ϕ1.atoms[j],use_coef=true)
                  for i=1:ϕ1.nterms,j=1:ϕ1.nterms)
        for i = 1:ϕ1.nterms
            ϕ1.atoms[i].coef /= sqrt(ϕ1_norm)
        end

        # update energy
        newenergy = .5*sum(dgradinnerprod(ϕ1.atoms[i],ϕ1.atoms[j],use_coef=true)
                        for i=1:ϕ1.nterms,j=1:ϕ1.nterms)+
                        sum(dinnerprod(ϕ1.atoms[i],
                        dprodgatoms(v.atoms[j],ϕ1.atoms[k]),use_coef=true)
                        for i=1:ϕ1.nterms,j=1:v.nterms,k=1:ϕ1.nterms)

        newμ = sqrt(-2.0*newenergy)

        # update solution
        ϕ0 = ϕ1

        energy,μ = newenergy,newμ

        println("energy = ",energy," μ = ", μ)
    end

    return energy,ϕ0
end

function tdse_exact(E,ϕ,x,v,t)
    """
    Exact solution to the TDSE corresponding to the moving atom.
    The solution to the TISE is represented as a Gaussian mixture ϕ
    with the orbital energy E.
    """
    ψ = exp.(-1.0im*(E+v*v/2.0)*t.+1.0im.*x.*v).*dwavefun_val(ϕ,x)
    return ψ
end


function main()
    E,ϕ = tise_solve(maxiter=20)
end

end # module
