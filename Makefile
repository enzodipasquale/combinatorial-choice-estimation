.PHONY: run_firms run_auction run_firms_mpi run_auction_mpi run_auction_old run_auction_mpi_old

run_auction:
	mpiexec -n 4 python3 -m applications.combinatorial_auction.run_estimation

quadsupermod_simulate_data:
	mpiexec -n 4 python3 -m applications.quadsupermod_demo.generate_simulated_data

run_quadsupermod:
	mpiexec -n 4 python3 -m applications.quadsupermod_demo.run_estimation

greedy_simulate_data:
	mpiexec -n 4 python3 -m applications.greedy_demo.generate_simulated_data

run_greedy:
	mpiexec -n 4 python3 -m applications.greedy_demo.run_estimation


linearknap_simulate_data:
	mpiexec -n 4 python3 -m applications.linearknapsack_demo.generate_simulated_data