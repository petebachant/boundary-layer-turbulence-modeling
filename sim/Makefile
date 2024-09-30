.PHONY: docker
docker:
	docker build -t blsim .
	docker inspect --format '{{.Id}}' blsim > Dockerfile.digest

BASE_CMD=docker run -it --rm -w /sim -v $(shell PWD):/sim blsim

.PHONY: run-docker
run-docker:
	${BASE_CMD}

.PHONY: mesh
mesh:
	${BASE_CMD} blockMesh | tee log.blockMesh

.PHONY: clean-all
clean:
	${BASE_CMD} python -c "import foampy; foampy.clean()"

.PHONY: clean-sim
clean-sim:
	rm -rf ./[1-9]* log.simpleFoam

.PHONY: run-sim
run-sim:
	${BASE_CMD} simpleFoam | tee log.simpleFoam

.PHONY: post-process
post-process:
	${BASE_CMD} postProcess -latestTime -func sample
