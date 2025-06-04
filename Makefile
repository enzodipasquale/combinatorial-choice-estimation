.PHONY: run_firms run_auction run_firms_mpi run_auction_mpi run_auction_old run_auction_mpi_old, \
	quadsupermod_simulate_data run_quadsupermod greedy_simulate_data run_greedy \
	linearknap_simulate_data run_linearknap

run_auction:
	mpiexec -n 6 python3 -m applications.combinatorial_auction.run_estimation

quadsupermod_simulate_data:
	mpiexec -n 6 python3 -m applications.quadsupermod_demo.generate_simulated_data

run_quadsupermod:
	mpiexec -n 6 python3 -m applications.quadsupermod_demo.run_estimation

greedy_simulate_data:
	mpiexec -n 6 python3 -m applications.greedy_demo.generate_simulated_data

run_greedy:
	mpiexec -n 6 python3 -m applications.greedy_demo.run_estimation

linearknap_simulate_data:
	mpiexec -n 6 python3 -m applications.linearknapsack_demo.generate_simulated_data

run_linearknap:
	mpiexec -n 6 python3 -m applications.linearknapsack_demo.run_estimation

run_firms:
	mpiexec -n 6 python3 -m applications.firms_export.run_estimation