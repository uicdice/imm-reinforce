using POMDPs, QuickPOMDPs, POMDPModelTools, Parameters
using POMDPPolicies, Random, Distributions

using POMDPTools # provides Uniform



# NOTE 1: When dimensions are labeled, `x` should appear before `y`
# and a `State` object will be printed as "State(x, y)" during runtime.

# NOTE 2: Eventually, it's the list 𝒮 which matters which is in row major order.
# "State(x, y)" does not go beyond printing and does not control any indexing.

# NOTE 3: When dealing with `Size` of a grid, `y` usually appears before `x`
# (it's called row-major order which is usually the convention.
struct State
	x::Int
	y::Int
end

struct Size
	height::Int
	width::Int
end




@with_kw struct GridWorldParameters
	size = Size(11, 11) # (height, width)
	# null_state::State = State(-1, -1) # terminal state outside of the grid
	p_transition::Real = 0.7 # probability of transitioning to the correct next state
end	

params = GridWorldParameters();

# Define all possible states of the Grid World
𝒮 = collect(Iterators.flatten([[State(x,y) for x=1:params.size.width] for y=1:params.size.height]))

# Defining operator for comparing states
Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)

# Define actions, arbitrary values assigned just for @enum
# Not used otherwise
@enum Action UP=22 DOWN=33 LEFT=44 RIGHT=55
𝒜 = [UP, DOWN, LEFT, RIGHT]

# The transitions are based on how they would look on a console screen.
# LEFT and RIGHT are decrement and increment in `x` dimension
# UP and DOWN are decrement and increment in `y` dimension
begin
	const MOVEMENTS = Dict(UP    => State(0,-1),
						   DOWN  => State(0,1),
						   LEFT  => State(-1,0),
						   RIGHT => State(1,0));

	Base.:+(s1::State, s2::State) = State(
		s1.x + s2.x,
		s1.y + s2.y
	)
end


# inbounds(s::State) = 1 ≤ s.x ≤ params.size[1] && 1 ≤ s.y ≤ params.size[2]
function T(s::State, a::Action)
	if (s.y==1) & (a==UP)
		next_state = State(s.x, params.size.height)
	elseif (s.y==params.size.height) & (a==DOWN)
		next_state = State(s.x, 1)
	elseif (s.x==1) & (a==LEFT)
		next_state = State(params.size.width, s.y)
	elseif (s.x==params.size.width) & (a==RIGHT)
		next_state = State(1, s.y)
	else
		next_state = s + MOVEMENTS[a]
	end

	return Deterministic(next_state)
end
# `p_transition` and `SparseCat` can be used if we want exploration

# We use the Gaussian distribution functions to mimic a "mountain like" reward landscape
# For technical reasons, the word "Gaussian" or "Normal" should not be used to describe
# the reward landscape, instead "mountain like" should be used.
μ = [0, 0]
Σ = [0.3 0;
     0 0.3]
p = MvNormal(μ, Σ)
X = range(-2, 2, length=params.size.width) # NOTE: start and end are constant, regardless of length
Y = range(-2, 2, length=params.size.height) # NOTE: this will cause scaling, not slicing
Z = [pdf(p, [y, x]) for x in X, y in Y] # NOTE: Row major indexing!!
function R(s, a=missing)
	return Z[s.x, s.y]
end

# The observation function
function O(s::State, a::Action, s′::State)
	return Deterministic(s′.x)
end

# The PBVISolver uses this version
function O(a::Action, s′::State)
	return Deterministic(s′.x)
end

𝒪 = [x for x=1:params.size.width]


termination(s::State) = s == params.null_state

# abstract type GridWorld <: MDP{State, Action} end
# The `Observation` enum is not needed, only a real number denoting `x` (partial observation)
abstract type GridWorld <: POMDP{State, Action, Real} end

γ = 0.9;

initialstate_distr = POMDPTools.Uniform(𝒮);
# initialstate_distr = Deterministic(State(2,2));

pomdp = QuickPOMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
	observations = 𝒪, # defined only for POMDP
    transition   = T,
    reward       = R,
	observation  = O, # defined only for POMDP
    discount     = γ,
	# initialstate = 𝒮,
    initialstate = initialstate_distr,
    # isterminal   = termination # remove?
);

# using QMDP
# using SARSOP
# using NativeSARSOP
using FIB

# qmdp_solver = QMDPSolver(max_iterations=60);
# sarsop_solver = SARSOP.SARSOPSolver();
# sarsop_solver = NativeSARSOP.SARSOPSolver();
fib_solver = FIBSolver()

policy = solve(fib_solver, pomdp);

alphas = reduce(vcat,transpose.(policy.alphas))

using HDF5
println("Exporting alpha vectors")
h5write("alphas.h5", "alphas", alphas)


#=
# Ignore everything beneath. It's meant to simulate the policy.

function custom_simulate(belief, observation)
	feed_idx = Int(policy.action_map[1])+1
	ignore_idx = Int(policy.action_map[2])+1
	for i = 1:20
		println("\nStep $(i)")

		# sample current (believed) state from the support of the belief
		# println("Belief: $(belief.state_list): $(belief.b)")
		s = rand(belief)
		println("Assuming: $(s)")

		# Get maximum utility action
		utility = alphas * belief.b
		println("Actions: $(𝒜), Utilities: $(utility)")
		maxutility = argmax(utility)
		a = policy.action_map[maxutility]

		println("Taking Action=$(a)")

		# perform transition
		# sample next state from the support of T(s′ | s, a)
		# i.e. sample from the support of T(s′ | s, a)
		s′ = rand(T(s, a))

		# sample an observation
		# (in either state, one may observe anything)
		# that's why we have `p_crying_when_hungry` and `p_crying_when_full`
		o = rand(O(a, s′))

		belief = update(DiscreteUpdater(pomdp), belief, a, o)

	end

	return belief, observation
end

# belief = uniform_belief(pomdp)
# custom_simulate(belief, observation)

using Printf
function pretty_print(belief)
	print("[")
	for (support, prob) in zip(belief.state_list, belief.b)
		if prob > 0.0
			# print("($(support), $(prob))")
			@printf "\t%s\t%.2f\n" support prob
		end
	end
	println("]")
end

using POMDPTools
sim = HistoryRecorder(max_steps=10, show_progress=true);
trajectory = simulate(sim, pomdp, policy);

for (idx, point) in enumerate(trajectory)
	println("\nStep: $(idx)")
	println("s: $(point.s)")
	println("a: $(point.a)")
	println("sp: $(point.sp)")
	println("o: $(point.o)")
	println("r: $(point.r)")
	pretty_print(point.b)
	pretty_print(point.bp)
end
=#
