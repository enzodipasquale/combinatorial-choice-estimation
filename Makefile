.PHONY: run_firms run_auction run_firms_mpi run_auction_mpi run_auction_old run_auction_mpi_old

run_auction:
	mpiexec -n 4 python3 -m applications.combinatorial_auction.run_estimation

run_firms_mpi:
	mpiexec -n 4 python3 -m applications.firms_export.run_estimation

greedy_simulate_data:
	mpiexec -n 4 python3 -m applications.greedy_demo.generate_simulated_data

run_greedy:
	mpiexec -n 4 python3 -m applications.greedy_demo.run_estimation