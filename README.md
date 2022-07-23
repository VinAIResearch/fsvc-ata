# Inductive and Transductive Few-Shot Video Classification via Appearance and Temporal Alignments
by [Khoi D. Nguyen](https://khoiucd.github.io), [Quoc-Huy Tran](https://cs.adelaide.edu.au/~huy/home.php), [Khoi Nguyen](https://www.khoinguyen.org), [Binh-Son Hua](https://sonhua.github.io), and [Rang Nguyen](https://rangnguyen.github.io)
> **Abstract**: 
We present a novel method for few-shot video classification, which performs appearance and temporal alignments. In particular, given a pair of query and support videos, we conduct appearance alignment via frame-level feature matching to achieve the appearance similarity score between the videos, while utilizing temporal order-preserving priors for obtaining the temporal similarity score between the videos. Moreover, we introduce a few-shot video classification framework that leverages the above appearance and temporal similarity scores across multiple steps, namely prototype-based training and testing as well as inductive and transductive prototype refinement. To the best of our knowledge, our work is the first to explore transductive few-shot video classification. Extensive experiments on both Kinetics and Something-Something V2 datasets show that both appearance and temporal alignments are crucial for datasets with temporal order sensitivity such as Something-Something V2. Our approach achieves similar or better results than previous methods on both datasets.

![teaser](teaser.png)

Details of our evaluation framework and benchmark results can be found in our paper:
```bibtex
@inproceedings{khoi2022ata,
    title={Inductive and Transductive Few-Shot Video Classification via Appearance and Temporal Alignments},
    author={Khoi D. Nguyen and Quoc-Huy Tran and Khoi Nguyen and Binh-Son Hua and Rang Nguyen},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2022}
}
```
**Please CITE** our paper when this repository is used to help produce published results or is incorporated into other software.
