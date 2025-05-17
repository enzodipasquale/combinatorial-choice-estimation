.PHONY: run_firms run_auction run_firms_mpi run_auction_mpi run_auction_old run_auction_mpi_old

run_auction:
	mpiexec -n 4 python3 -m applications.combinatorial_auction.run_estimation

run_quad-supermod:
	mpiexec -n 4 python3 -m applications.quad-supermod_demo.run_estimation

quad_supermod_simulate_data:
	mpiexec -n 4 python3 -m applications.quad-supermod_demo.generate_simulated_data

greedy_simulate_data:
	mpiexec -n 4 python3 -m applications.greedy_demo.generate_simulated_data

run_greedy:
	mpiexec -n 4 python3 -m applications.greedy_demo.run_estimation