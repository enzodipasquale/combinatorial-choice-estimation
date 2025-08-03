all_rowgen_tests: supermod_test greedy_test single_test knap_test





greedy_experiment:
	mpirun -n 10 python experiments/row_generation_greedy_experiment.py


greedy:
	mpirun -n 10 python benchmarking/greedy/experiment.py
supermod:
	mpirun -n 10 python benchmarking/supermod/experiment.py
knapsack:
	mpirun -n 10 python benchmarking/knapsack/experiment.py


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

all_benchmarks: greedy_benchmark supermod_benchmark


