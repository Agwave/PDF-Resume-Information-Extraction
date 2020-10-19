## 0. 摘要
参加了天池的一个pdf简历信息提取的比赛，这里进行回顾、整理和分享

赛题要求从pdf简历中提取出信息，比如说名字，籍贯等。这里搭建了一个BiLSTM-CRF模型，能够从PDF简历中提取出所需的信息。

模型的线上得分是0.727，排名 21/1200+

## 1. 赛题相关

**模型目标**：pdf简历 --> 类别信息 

## 2. 思路
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407184911162.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxODA1NTEx,size_16,color_FFFFFF,t_70)
使用python库**pdfminer**，将pdf简历中的文本提取出来。利用json标注文件，对提取出来的文本进行匹配和**BIO标注**，每一个字对应一个标注。最后，将标注后的文本送到BiLSM-CRF模型中进行训练。
## 3. BiLSTM-CRF 模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200417140041730.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxODA1NTEx,size_16,color_FFFFFF,t_70)
将文本中的每个字进行one-hot编码，经过Embedding层后，每一个字对应一个**字向量**，所以文本可以用一个矩阵表示。将**文本矩阵**输入BiLSTM层，输出中每一个字会对应一个类别概率向量，此类别概率向量表示了该字属于各个类别的概率。所以所有字属于各个类别的概率可以用一个**类别概率矩阵**表示。将此类别概率矩阵输入CRF层，即可得到**得分最高的文本标注序列**。

此处留一个pytorch官方的BiLSTM-CRF教程链接：
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#

## 4. 代码地址
**https://github.com/Agwave/PDF-Resume-Information-Extraction**
## 5. 不足
1. 没有利用外部文本来训练语言模型。语言模型的文本只利用了训练集的pdf中的文本。
2. 只使用了字嵌入。中文文本的话还可以结合词嵌入。

