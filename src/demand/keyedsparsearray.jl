struct KeyedSparseArray{T, N} <: AbstractArray{T, N}
    data::SparseArray{T, N}
    keys::Vector{CartesianIndex{N}}
end
KeyedSparseArray(a::SparseArray) = KeyedSparseArray(a, collect(nonzero_keys(a)))

SparseArrayKit.nonzero_keys(sa::KeyedSparseArray) = sa.keys

# elementary getindex and setindex!
@inline function Base.getindex(a::KeyedSparseArray{T, N}, I::CartesianIndex{N}) where {T, N}
    @boundscheck checkbounds(a.data, I)
    return get(a.data.data, I, zero(T))
end
Base.@propagate_inbounds function Base.getindex(
        a::KeyedSparseArray{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    return getindex(a.data, CartesianIndex(I))
end

@inline function Base.setindex!(a::KeyedSparseArray{T, N}, v, I::CartesianIndex{N}) where {T, N}
    @boundscheck checkbounds(a.data, I)
    a.data.data[I] = v # require that key already was present
    return v
end
Base.@propagate_inbounds function Base.setindex!(
        a::KeyedSparseArray{T, N},
        v, I::Vararg{Int, N}
    ) where {T, N}
    return setindex!(a.data, v, CartesianIndex(I))
end

import InteractiveUtils
import VectorInterface # needed for methodswith
import TensorOperations as TO

function _generate_method_overrides()
    for m in InteractiveUtils.methodswith(SparseArray)
        m.name in [:nonzero_keys, :getindex, :setindex!] && continue
        tv, sig, _... = Base.arg_decl_parts(m)
        (_, fname), rest = Iterators.peel(sig) # rest is iterator of (argname, argtype) as Strings
        occursin("SparseArray", fname) && continue
        rest = collect(rest)
        fname = occursin(fname, "*/\\_-+") ? ":$fname" : fname
        fname = fname == "==" ? ":(==)" : fname
        sparse_array_args = findall(x -> occursin(r"^(SparseArrayKit\.)?SparseArray($|{)", x), last.(rest))
        already_defined = any(startswith.(last.(rest), "KeyedSparseArray"))
        already_defined && continue
        for i in sparse_array_args
            arg_strings = map(eachindex(rest)) do j
                if j == i
                    "$(rest[j][1] == "" ? "myvar" : rest[j][1])::$(replace(rest[j][2], "SparseArrayKit.SparseArray" => "KeyedSparseArray"))"
                else
                    "$(rest[j][1])$(rest[j][2] == "" ? "" : "::$(rest[j][2])")"
                end
            end
            caller_strings = map(eachindex(rest)) do j
                if j == i
                    "$(rest[j][1] == "" ? "myvar" : rest[j][1]).data"
                else
                    "$(rest[j][1] == "" ? "$(rest[j][2])()" : rest[j][1])"
                end
            end
            arglist = join(arg_strings, ", ")
            callerlist = join(caller_strings, ", ")
            tvlist = isempty(tv) ? "" : "where {$(join(tv, ", "))}"
            og_definition = readlines(string(m.file))[m.line]
            modstring = match(Regex(" ?(\\w*)\\.?:?\\(?\\Q$fname\\E\\("), og_definition)
            modstring = isnothing(modstring) ? "SparseArrayKit" : modstring.captures[1]
            modstring = modstring == "" ? "SparseArrayKit" : modstring
            funcmacros = match(r"^(\w*\.?@\w+\s+)+\s*function", og_definition)
            funcmacros = isnothing(funcmacros) ? "" : join(funcmacros.captures, " ")
            definition = Meta.parse("""
                $funcmacros function $(modstring).$(fname)($(arglist)) $(tvlist)
                    return $(modstring).$(fname)($(callerlist))
                end
            """)
            @eval @__MODULE__() $definition
        end
    end
end
