# ML_HSE - A repository for ML course of HSE "Data Science"
Here are some interesting problems given as course home assignments. Some of them are quite applicable. Feel free to read, understand the insights and use.<br>

### Interesting Linear Regression Problem
#### Setting
Consider two sets of points $A$ and $B$ in $\mathbb{R}^2$ space. Linear Regression fit on $A$  is $y = k_1x + y_1$, $k_1 > 0$; and Linear Regression fit on $B$ is $y = k_2x + y_2$, $k_2 > 0$.
#### Quesion
Is it true that Linear Regression fit on $A \cup B$ is $y = k_3x + y_3$ with $k_3 > 0$. If yes, then prove it. Provide examples otherwise.<br>
You can look through reg_problem.ipynb file to see some interesting cases and solutions to this problem.<br>

### K Nearest Neighbours Implementation
Here you can find my implementation of KNN classifier with Euclidian metric (the code may be generalized to any other). Implementation is pretty much straightforward and uses brute force.<br>
Some crucial classification metrics are implemented as well. These include:
<ul>
<li> Accuracy</li>
<li> True Positive Rate</li>
<li> False Positive Rate</li>
<li> Precision</li>
<li> Recall</li>
<li> F1 score </li>
</ul>

See the implementation in KNN.py file

### Mutual Information and Differential Mutual Information Score Implementation
#### Mutual Information Score
Consider the following: X is a set of predicted classes(labels), Y is a set of true classes. $p_X(x)$ is probability of point to be predicted as class X=x, $p_Y(y)$ is probability of point to be of class Y=y, $p_{X,Y}(x,y)$ is probability of predicting class to be x and being y in reality ($p_X(x)$ and $p_Y(y)$ are marginal distributions of predicted and true classes, $p_{X,Y}(x,y)$ is their joint distribution). Then their mutual information is calculated as follows.<br>
```math
I(X,Y) = \sum_{x \in X} \sum_{y \in Y} p_{X,Y}(x,y) \log \frac{p_{X,Y}(x,y)}{p_X(x)p_Y(Y)} = // p_X(x) = \frac{|x|}{N}, p_y = \frac{|y|}{N}, p_{X,Y}(x,y) = \frac{|x \cap y|}{N}// = \sum_{x \in X} \sum_{y \in Y} \frac{|x \cap y|}{N} \log \frac{N |x \cap y|}{|x||y|}
```
Given predcited labels and corresponding true labels mutual information can be easily calculated using contingency matrix.<br>

#### Differentiral Mutual Information Score
Now consider we have predicted probabilities of labels. In this case the formula remains the same, but now $p_X(x)$ can not be calculated as $\frac{|x|}{N}$. Instead it is calculated as $p_X(x) = \sum_i p_i p_i(x) = \sum_i \frac{p_i(x)}{N} = \mathbb{E}_i[ p_i(x) ]$ , where $p_i$ is probability of choosing point $i$, $p_i(x)$ is probability of point $i$ to be predicted as class X=x. Here to evaluate the joint distribution we calculate predicted labels as those with greater probability.<br>

See the implementation with examples in mut_info.ipynb file. See also mut_info_scores file for pure function scripts.
