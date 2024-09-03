.PHONY: docker
docker:
	docker build -t blsim .

BASE_CMD=docker run -it -w /root -v $(shell PWD):/root blsim

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
	rm -rf ./1* log.simpleFoam

.PHONY: run-sim
run-sim:
	${BASE_CMD} simpleFoam | tee log.simpleFoam
