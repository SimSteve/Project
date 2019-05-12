# Project
Yishay Asher & Steve Gutfreund <br>
final project for Bachelor's in Computer Science <br>
Securing Machine Learning Models from Adversarial Attacks <br> <br>
Article: https://arxiv.org/pdf/1809.01715.pdf

stuff we need to remember:
1. we splitted the predict function in Models.py , and added encrypt_predict (=> made changes in visualize_attack and l2_attack)
2. check targetted
3. permutation can be represented as a linear layer, that's why for an attacker in a white box scenario can easily attack, but training is not trivial, because this layer is not being updated during the training
