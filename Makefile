.PHONY: run_firms run_auction run_firms_mpi run_auction_mpi

run_firms:
	python3 -m applications.firms_export.run_estimation

run_auction:
	python3 -m applications.combinatorial_auction.run_estimation

run_firms_mpi:
	mpiexec -n 4 python3 -m applications.firms_export.run_estimation

run_auction_mpi:
	mpiexec -n 4 python3 -m applications.combinatorial_auction.run_estimation

run_auction_mpi_old:
	mpiexec -n 4 python3 GMM_quad_old/job-main.py

