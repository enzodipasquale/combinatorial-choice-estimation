all_rowgen_tests: supermod_test greedy_test single_test knap_test

supermod_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_quadsupermodular.py | cat

greedy_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_greedy.py | cat

single_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_plain_single_item.py | cat

knap_test:
	mpirun -n 2 pytest -s --log-cli-level=INFO bundlechoice/tests/test_row_generation_linear_knapsack.py | cat 