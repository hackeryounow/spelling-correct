

![image-20230610202141505](https://pggo.oss-cn-beijing.aliyuncs.com/img/image-20230610202141505.png)

给定一个单词，要计算的是从候选的单词选择最大的p(c|s)，假设给定的句子是$x_1, x_2, x_3, x_4, x_5$，以$x_3$为s单词，计算p(c|s)，

p(s|c)表示c拼错的单词为s的概率，通过spelling_error.txt计算

`scold: schold, skold`

c为单词schold，拼错的概率为1/2

$p(c)=p(x_2|c)p(x_4|c)$ 表示c可作为$x_3$的替换的概率

由于直接乘会使数变得特别小，所以对其取log

$argmax_{c \in candidates} p(s|c)*p(c)=argmax_{c \in candidates} log(p(s|c)*p(c)) = argmax_{c \in candidates} log(p(s|c))+ log(p(c))$ 

### Reuters Corpora

Reuters Corpora (RCV1, RCV2, TRC2) 是一个英文新闻语料数据，包括大量的英文新闻及分类标注。[更多](https://languageresources.github.io/2018/06/16/%E5%88%98%E6%99%93_Reuters%20Corpora%20%E8%8B%B1%E6%96%87%E6%96%B0%E9%97%BB%E6%95%B0%E6%8D%AE/)

```python
import nltk
nltk.download('reuters')
```

[下载地址](https://github.com/nltk/nltk_data)

将packages下的所有文件夹放到Anoconda的文件夹下面`D:\Anaconda3\nltk_data`

### 具体步骤

（1）找寻单词编辑距离

例如：编辑距离为1的操作，插入一个字符，删除一个字符，替换一个字符

```python
def generate_candidates(word):
    """
    word: 给定的输入（错误的输入） 
    返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl... 
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app
    
    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz' 
    
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    # insert操作
    inserts = [L+c+R for L, R in splits for c in letters]
    # delete
    deletes = [L+R[1:] for L,R in splits if R]
    # replace
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    # 去重
    candidates = set(inserts+deletes+replaces)
    
    # 过来掉不存在于词典库里面的单词
    return [word for word in candidates if word in vocab] 
    
generate_candidates("apple")
```

（2）单词过滤（）

停用词、出现频率很低的词汇过滤掉

出现频率特别低的词汇对分析作用不⼤，所以一般也会去掉。把停用词、出现频率低的词过滤之后，即可以得到一个我们的词典库。

（3）Steamming

```sh
# 单词类似，怎么合并
went, go, going
fly, flies, 
deny, denied, denying
fast, faster, fastest
```

参考算法：https://tartarus.org/martin/PorterStemmer/java.txt

（4）构建语言模型

```python
from nltk.corpus import reuters

def construct_lang_model():
    term_count = {}
    bigram_count = {}
    categories = reuters.categories()
	corpus = reuters.sents(categories=categories)
    for doc in corpus:
        # 增加起始字符
        doc = ['<s>'] + doc
        for i in range(0, len(doc)-1):
            # bigram: [i,i+1]
            term = doc[i]
            bigram = doc[i:i+2]

            if term in term_count:
                term_count[term]+=1
            else:
                term_count[term]=1
            bigram = ' '.join(bigram)
            if bigram in bigram_count:
                bigram_count[bigram]+=1
            else:
                bigram_count[bigram]=1
	return term_count, bigram_count
```

（4）统计用户输错的概率

```python
def channel_prob(spell_error_corpus_file):
    channel_prob = {}
    for line in open(spell_error_corpus_file):
        items = line.split(":")
        correct = items[0].strip()
        mistakes = [item.strip() for item in items[1].strip().split(",")]
        channel_prob[correct] = {}
        for mis in mistakes:
            channel_prob[correct][mis] = 1.0/len(mistakes)
    return channel_prob
```

（5）测试

```python
def spelling_correct(sentence, vocab, V):
    if sentence.endswith("."):
        sentence = sentence[::-1].replace(".", " .", 1)[::-1]
    sentence = sentence.replace(",", " .")
    words = sentence.split()
    corrected_words = []
    origin_words = []
    for idx in range(len(words) - 1):
        word = words[idx]
        if word in vocab:
            continue
        candidates = generate_candidates(word)
        if len(candidates) == 0:
            continue
        probs = []
        for candidate in candidates:
            prob = 0
            # 计算channel probility
            if candidate in channel_proba and word in channel_proba[candidate]:
                prob += np.log(channel_proba[candidate][word])
            else:
                prob += np.log(0.0001)

            # 计算语言模型
            bigram_key = ""

            if idx == 0:
                bigram_key += '<s>'
            else:
                bigram_key += words[idx - 1]

            bigram_key += " " + candidate
            if bigram_key in bigram_count.keys():
                prob += np.log((bigram_count[bigram_key] + 1.0) / (term_count[candidate] + V))
            else:
                prob += np.log(1.0 / V)

            bigram_key += candidate + " " + words[idx + 1]
            if bigram_key in bigram_count.keys():
                prob += np.log((bigram_count[bigram_key] + 1.0) / (term_count.get(candidate, 0) + V))
            else:
                prob += np.log(1.0 / V)
            probs.append(prob)
        max_id = probs.index(max(probs))
        corrected = candidates[max_id]
        if corrected != word:
            origin_words.append(word)
            corrected_words.append(corrected)
    return origin_words, corrected_words
```

