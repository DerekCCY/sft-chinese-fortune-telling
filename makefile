run:
	python -m data_generation.data_gen_pipeline

births:
	python -m data_generation.gen_births --count 500

charts:
	python -m data_generation.gen_charts
