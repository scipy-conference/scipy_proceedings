all: papers

.PHONY: papers
papers: papers/*
	python publisher/build_paper.py $?

