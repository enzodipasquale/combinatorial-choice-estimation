all_rowgen_tests: supermod_test greedy_test single_test knap_test

supermod_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_quadsupermodular.py | cat

greedy_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_greedy.py | cat

single_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_plain_single_item.py | cat

knap_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_linear_knapsack.py | cat 

greedy:
	mpirun -n 8 python experiments/greedy/experiment.py

supermod:
	mpirun -n 8 python experiments/quadsupermod/experiment.py


greedy_benchmark:
	@for i in $(shell seq 1 100); do \
		echo "—— Run $$i ——"; \
		mpirun -n 8 python experiments/greedy/experiment.py || exit 1; \
		wolframscript -file /Users/enzo-macbookpro/MyProjects/score-estimator/greedy/greedy.wl || exit 1; \
	done