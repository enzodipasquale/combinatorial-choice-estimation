.PHONY: run_firms run_auction

run_firms:
	python3 -m applications.firms_export.run_estimation

run_auction:
	python3 -m applications.combinatorial_auction.run_estimation

