.PHONY: docker
docker:
	docker build -t blsim .


.PHONY: run-docker
run-docker:
	docker run -it -w /root -v $(shell PWD):/root blsim


.PHONY: clean
clean:
	@docker run -it -w /root -v $(shell PWD):/root blsim python -c "import foampy; foampy.clean()"


.PHONY: run-sim
run-sim:
	docker run -it -w /root -v $(shell PWD):/root blsim /bin/sh -c "blockMesh | tee log.blockMesh && simpleFoam | tee log.simpleFoam"
