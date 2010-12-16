all: papers

.PHONY: papers
papers: papers/*
	./build_paper.py $?
