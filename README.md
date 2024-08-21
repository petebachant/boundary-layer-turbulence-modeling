# Boundary layer turbulence modeling

Attempting to generate and validate
a new turbulence model from boundary layer DNS.

## Questions

1. This is the first question.
2. This is the second question.

This is an unordered list:
* This is an unordered list item.
* This is another.

## Workflow

```mermaid
flowchart TD
	node1["build-paper"]
	node2["compute-coeffs"]
	node3["data/jhtdb-transitional-bl/time-ave-profiles.h5.dvc"]
	node4["extract-jhtdb-stats"]
	node5["plot-time-ave-profiles"]
	node6["run-rans-sim"]
	node2-->node6
	node3-->node5
	node4-->node2
	node5-->node1
	node6-->node5
```
