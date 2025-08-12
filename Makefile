
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
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/greedy/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/greedy.wl || exit 1; \
	done

supermod_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/supermod/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/supermod/supermod.wl || exit 1; \
	done

knapsack_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/knapsack/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/knapsack/knapsack.wl || exit 1; \
	done

plain_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 10 python benchmarking/plain_single_item/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/plain_single_item/plain_single_item.wl || exit 1; \
	done

all_benchmarks: greedy_benchmark supermod_benchmark plain_benchmark




auction:
	mpirun -n 10 python applications/combinatorial_auction/run_estimation.py








