# Standalone scripts

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by
> Pete Bachant.

Scripts here are **deliberately not pipeline stages**. They are kept out
because they need credentials, network access to a third-party service, or a
dependency the rest of the project should not have to install.

Nothing in the reproducible pipeline may depend on them. If an output of one of
these is needed downstream, commit the output as a tracked dataset and have the
pipeline read that, so `calkit run` stays runnable by someone with only the
repository.

| script | why it is out of the pipeline |
|---|---|
| `fetch-jhtdb-gradients.py` | Needs a JHTDB access token and `pyJHTDB`, which will not build against numpy >= 1.24. Its output is not currently used by anything. |

## Running one

These have a declared calkit environment even though they are not stages, so
the token-gated path is reproducible rather than folklore:

```sh
calkit xenv -n py-jhtdb -- python scripts/standalone/fetch-jhtdb-gradients.py
```

`py-jhtdb` (`envs/jhtdb/`) is pinned to Python 3.11 and numpy < 1.24 because
`pyjhtdb` builds from source against APIs removed in numpy 1.24. Because no
pipeline stage uses that environment, no stage depends on its lock, so it can
never invalidate a result.
