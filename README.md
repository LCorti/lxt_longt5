# Tweaked LXT for T5-style models

> For the original repository and documentation, visit [https://github.com/rachtibat/LRP-eXplains-Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers). Consider citing they work!

# Minimal Dependencies
```
torch==2.7.1
transformers==4.53.3
accelerate==1.10.0
tabulate==0.9.0
matplotlib==3.10.1
zennit==1.0.0
```

**Note**: to visualise the outputs using the existing `pdf_heatmap` function, you need to have `pdflatex` or `xelatex` installed on your machine.

---

# Acknowledgements

## License
[LXT](https://github.com/rachtibat/LRP-eXplains-Transformers) is licensed under the BSD-3 Clause License. It can be used free of charge for personal and scientific purposes.

## Citation
```
@InProceedings{pmlr-v235-achtibat24a,
  title = {{A}ttn{LRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {135--168},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```

## Zennit
The code is heavily inspired by [Zennit](https://github.com/chr5tphr/zennit), a tool for LRP attributions in PyTorch using hooks. Zennit is 100% compatible with the **explicit** version of LXT and offers even more LRP rules.
