all: papers

.PHONY: papers
papers: papers/*
	./mk_scipy_paper.py $?


