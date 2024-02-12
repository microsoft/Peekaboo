# Datasets and Evaluation

## SSv2-ST (SSv2 Spatio-Temporal dataset)

### Pre-processing
Our pre-processing pipeline is described here. We first extract the first noun chunk of the caption using Spacy. Then this subject is fed into Owl-ViT-L to obtain bounding boxes. If there are 0 bounding boxes corresponding to a subject, we use the next caption from the dataset. If there are atleast two bounding boxes, we interpolate bounding boxes for the missing frames linearly. The dataset downloading is a bit complex, you need to follow the instructions [here](https://github.com/MikeWangWZHL/Paxion#dataset-setup). Download the dataset and run `generate_ssv2_st.py`.

## Interactive Motion Control - IMC
We generate bounding boxes for this dataset using the `generate_imc.py` file. The prompts are in `custom_prompts.csv` and `filtered_prompts.csv`.

For more details regarding the datasets and evaluation strategy, please refer to the [Peekaboo paper](https://arxiv.org/abs/2312.07509).
