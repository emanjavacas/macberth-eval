# MacBERTh

This repository contains the code used to evaluate the historical BERT model (MacBERTh) trained on more than 3B tokens of historical English across 5 centuries. Publication detailing training, data sources and evaluation is forthcoming. 

In order to use the model, you can use the transformers API using MacBERThs online path: `emanjavacas/MacBERTh`.

```python
>>> from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
>>> path = 'emanjavacas/MacBERTh'
>>> model = AutoModelForMaskedLM.from_pretrained(path)
>>> tokenizer = AutoTokenizer.from_pretrained(path)
>>> pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> s = 'For we do it not actuallye in dede, but [MASK] in a misterye.'
>>> pipe(s)

[{'sequence': 'for we do it not actuallye in dede, but onely in a misterye.',
  'score': 0.4811665117740631,
  'token': 1656,
  'token_str': 'onely'},
 {'sequence': 'for we do it not actuallye in dede, but only in a misterye.',
  'score': 0.36467066407203674,
  'token': 1200,
  'token_str': 'only'},
 {'sequence': 'for we do it not actuallye in dede, but as in a misterye.',
  'score': 0.034449998289346695,
  'token': 878,
  'token_str': 'as'},
 {'sequence': 'for we do it not actuallye in dede, but rather in a misterye.',
  'score': 0.027537666261196136,
  'token': 1754,
  'token_str': 'rather'},
 {'sequence': 'for we do it not actuallye in dede, but yet in a misterye.',
  'score': 0.023396290838718414,
  'token': 1126,
  'token_str': 'yet'}]
 ```

The code is model-agnostic relying on the `huggingface` interface to transformers models. We do, however, rely on OED data for evaluation which we cannot release due to it not being freely available. Still, the code should be illustrative about the approach taken for the different tasks, and should be easy to adapt to datasets similar to the OED.

