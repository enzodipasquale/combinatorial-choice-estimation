
greedy_experiment:
	mpirun -np 1 python3 experiments/experiment_inversion_greedy.py

supermod_experiment:
	mpirun -np 10 python3 experiments/experiment_inversion_supermod.py


greedy:
	mpirun -n 10 python benchmarking/greedy/experiment.py
supermod:
	mpirun -n 10 python benchmarking/supermod/experiment.py
knapsack:
	mpirun -n 10 python benchmarking/knapsack/experiment.py
plain:
	mpirun -n 1 python benchmarking/plain_single_item/experiment.py


greedy_benchmark:
	@echo "Cleaning up previous results..."
	@rm -f benchmarking/greedy/results.csv
	@rm -f /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/AddDrop/estimation_results.csv
	@echo "Starting fresh benchmark runs..."
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/greedy/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/AddDrop/greedy.wl || exit 1; \
	done
	python benchmarking/greedy/benchmarking.py

supermod_benchmark:
	@echo "Cleaning up previous results..."
	@rm -f benchmarking/supermod/results.csv
	@rm -f /Users/enzo-macbookpro/MyProjects/score-estimator/supermod/AddDrop/estimation_results.csv
	@echo "Starting fresh benchmark runs..."
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/supermod/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/supermod/AddDrop/supermod.wl || exit 1; \
	done
	python benchmarking/supermod/benchmarking.py

knapsack_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/knapsack/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/knapsack/Swap/knapsack.wl || exit 1; \
	done
	python benchmarking/knapsack/benchmarking.py

plain_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/plain_single_item/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/plain_single_item/plain_single_item.wl || exit 1; \
	done

all_benchmarks: greedy_benchmark supermod_benchmark




auction:
	mpirun -n 10 python applications/combinatorial_auction/run_estimation.py

firms:
	mpirun -n 10 python applications/firms_export/run_estimation.py







