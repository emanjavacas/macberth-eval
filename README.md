# MacBERTh

This repository contains the code used to evaluate the historical BERT model (MacBERTh) trained on more than 3B of historical English across 5 centuries.
The code is model agnostic relying on the `huggingface` interface to transformers models, but does rely on OED data which we cannot release due to it not being freely available. Still, the code should be illustrative about the approach taken, and should be easy to adapt to datasets similar to the OED.

