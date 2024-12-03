module quantumSim

using LinearAlgebra
export QSim, WeightedKet, State, x!, t!, tdg!, cx!, h!


struct QSim{T <: Complex}
	n::Int # number of qubits
	m::Int # number of classical bits
	ops::Array{Array{T, 2}, 1}
	measure_low::Int
	measure_high::Int
end

function QSim(
	n::Int,
	m::Int,
	measure_low::Int = 1,
	measure_high::Int = n,
)
	ops = Array{Complex, 2}[]
	return QSim(n, m, ops, measure_low, measure_high)
end

struct WeightedKet{T <: Complex}
	amp::Vector{T} # amplitude
	ket::BitArray # |x>
end

function WeightedKet(
	n::Int,
	amp::Complex = 1.0 + 0.0im,
	ket::BitArray = falses(n),
)
	@assert length(ket) == n
	return WeightedKet([amp], ket)
end

function bitarray_to_index(bits::BitArray)
	return foldl((acc, b) -> acc * 2 + b, bits, init = 0) + 1
end

function index_to_bitarray(value::Int, n::Int)
    @assert value >= 1 && value <= 2^n
	bits = reverse(digits(value - 1, base = 2))
	return BitArray(vcat(zeros(Int, n - length(bits)), bits))
end

struct State{T <: Complex, TT <: BitArray}
	n::Int
	amps::Vector{T}
	kets::Vector{TT}
end

function State(n::Int, wks::Vector{WeightedKet{T}}=WeightedKet{ComplexF64}[]) where T <: Complex
    if isempty(wks)
        return State(n, zeros(T, 2^n), Vector{BitArray}(undef, 2^n))
    end
	@assert all(length(wk.ket) == n for wk in wks)
	"All WeightedKet objects must have the same n number of qubits."
	amps = zeros(T, 2^n)
	kets = Vector{BitArray}(undef, 2^n)
	for wk in wks
		idx = bitarray_to_index(wk.ket)
		amps[idx] = wk.amp[1]
		kets[idx] = wk.ket
	end
	return State(n, amps, kets)
end

function state(wks::Vector{WeightedKet{T}}) where T <: Complex
	if isempty(wks)
		throw(ArgumentError("Input vector wks cannot be empty."))
	end
	if any(length(wk.ket) != length(wks[1].ket) for wk in wks)
		throw(ArgumentError("All WeightedKet objects must have the same number of qubits."))
	end
	state = zeros(T, 2^length(wks[1].ket))
	for wk in wks
		idx = bitarray_to_index(wk.ket)
		state[idx] = wk.amp[1]
	end
	return state
end

include("gates.jl")

end
