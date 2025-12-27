
greedy_experiment:
	mpirun -np 10 python3 experiments/experiment_inversion_greedy.py

supermod_experiment:
	mpirun -np 10 python3 experiments/experiment_inversion_supermod.py


greedy:
	mpirun -n 10 python3 experiments/_deprecated/benchmarking_with_score/greedy/experiment.py
supermod:
	mpirun -n 10 python3 experiments/_deprecated/benchmarking_with_score/supermod/experiment.py
knapsack:
	mpirun -n 10 python3 experiments/_deprecated/benchmarking_with_score/knapsack/experiment.py
plain:
	mpirun -n 1 python3 experiments/_deprecated/benchmarking_with_score/plain_single_item/experiment.py


greedy_benchmark:
	@echo "Cleaning up previous results..."
	@rm -f benchmarking/greedy/results.csv
	@rm -f /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/AddDrop/estimation_results.csv
	@echo "Starting fresh benchmark runs..."
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python3 benchmarking/greedy/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/AddDrop/greedy.wl || exit 1; \
	done
	python3 benchmarking/greedy/benchmarking.py

supermod_benchmark:
	@echo "Cleaning up previous results..."
	@rm -f benchmarking/supermod/results.csv
	@rm -f /Users/enzo-macbookpro/MyProjects/score-estimator/supermod/AddDrop/estimation_results.csv
	@echo "Starting fresh benchmark runs..."
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python3 benchmarking/supermod/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/supermod/AddDrop/supermod.wl || exit 1; \
	done
	python3 benchmarking/supermod/benchmarking.py

knapsack_benchmark:
	@echo "Cleaning up previous results..."
	@rm -f benchmarking/knapsack/results.csv
	@rm -f /Users/enzo-macbookpro/MyProjects/score-estimator/knapsack/Swap/estimation_results.csv
	@echo "Starting fresh benchmark runs..."
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python3 benchmarking/knapsack/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/knapsack/Swap/knapsack.wl || exit 1; \
	done
	python3 benchmarking/knapsack/benchmarking.py

plain_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python3 benchmarking/plain_single_item/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/plain_single_item/plain_single_item.wl || exit 1; \
	done

all_benchmarks: greedy_benchmark supermod_benchmark knapsack_benchmark




auction:
	mpirun -n 10 python3 applications/combinatorial_auction/run_estimation.py

firms:
	mpirun -n 10 python3 applications/firms_export/run_estimation.py







