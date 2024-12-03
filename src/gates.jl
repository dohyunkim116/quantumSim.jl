function h!(i::Int, s::State{T}) where T
    @assert i <= s.n && i > 0 "Invalid qubit index"
    stemp = State(s.n);
    mask_amp = s.amps .!= zero(eltype(s.amps))
    indices_amp = findall(mask_amp)
    for l in indices_amp, k in [0, 1]
        temp = similar(s.kets[l]);
        temp .= s.kets[l];
        temp[i] = k;
        idx = quantumSim.bitarray_to_index(temp)    
        stemp.amps[idx] += s.amps[l] * (-one(T))^(k * s.kets[l][i]) / sqrt(2)
    end
    fillket!(stemp)
    s.amps .= stemp.amps
    s.kets .= stemp.kets
end

function fillket!(s)
    mask_amp = s.amps .!= zero(eltype(s.amps))
    indices_amp = findall(mask_amp)
    s.kets[indices_amp] .= quantumSim.index_to_bitarray.(indices_amp, Ref(s.n))
end

function _x!(i::Int, ket::BitArray)
    ket[i] = !ket[i]
end

function x!(i::Int, s::State)
    @assert i <= s.n && i > 0
    stemp = State(s.n);
    mask = s.amps .!= zero(eltype(s.amps))
    for l in findall(mask)
        temp = similar(s.kets[l]);
        temp .= s.kets[l];
        _x!(i, temp)
        idx = quantumSim.bitarray_to_index(temp)
        stemp.amps[idx] = s.amps[l]
    end
    fillket!(stemp)
    s.amps .= stemp.amps
    s.kets .= stemp.kets
end

function t!(i::Int, s::State)
    @assert i <= s.n && i > 0
    mask_amp = s.amps .!= zero(eltype(s.amps))
    mask_assn = [isassigned(s.kets, i) for i in eachindex(s.kets)]
    mask_ket = [b == 1 ? s.kets[l][i] : false for (l, b) in enumerate(mask_assn)]
    mask = mask_amp .& mask_ket
    s.amps[mask] .*= exp(1.0im * π / 4)
end

function tdg!(i::Int, s::State)
    @assert i <= s.n && i > 0
    mask_amp = s.amps .!= zero(eltype(s.amps))
    mask_assn = [isassigned(s.kets, i) for i in eachindex(s.kets)]
    mask_ket = [b == 1 ? s.kets[l][i] : false for (l, b) in enumerate(mask_assn)]
    mask = mask_amp .& mask_ket
    s.amps[mask] .*= exp(-1.0im * π / 4)
end

function cx!(i::Int, j::Int, s::State)
    @assert i <= s.n && i > 0
    @assert j <= s.n && j > 0
    @assert i != j
    stemp = State(s.n);
    mask_amp = s.amps .!= zero(eltype(s.amps))
    mask_assn = [isassigned(s.kets, i) for i in eachindex(s.kets)]
    mask_ket = [b == 1 ? s.kets[l][i] : false for (l, b) in enumerate(mask_assn)]
    mask = mask_amp .| mask_ket
    for l in findall(mask)
        temp = similar(s.kets[l]);
        temp .= s.kets[l];
        temp[i] == true ? temp[j] = !temp[j] : nothing
        idx = quantumSim.bitarray_to_index(temp)
        stemp.amps[idx] = s.amps[l]
    end
    fillket!(stemp)
    s.amps .= stemp.amps
    s.kets .= stemp.kets
end

