---
title: "Welcome to corpus"
id: welcome
url: https://github.com/example/corpus
---

# Welcome

This is a tiny sample corpus you can ingest to verify the pipeline works end-to-end before pointing `corpus` at your real archive.

Try:

```sh
corpus-ingest --source sample -v
corpus-query "what is corpus?"
corpus-query "retrieval pipeline architecture"
```

Each `.md` file in `examples/sample_corpus/notes/` becomes one document.
