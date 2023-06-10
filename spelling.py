from nltk.corpus import reuters
import numpy as np

vocab = set([line.rstrip() for line in open('vocab.txt')])


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

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # insert操作
    inserts = [L + c + R for L, R in splits for c in letters]
    # delete
    deletes = [L + R[1:] for L, R in splits if R]
    # replace
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    candidates = set(inserts + deletes + replaces)

    # 过来掉不存在于词典库里面的单词
    return [word for word in candidates if word in vocab]


def construct_lang_model():
    term_count = {}
    bigram_count = {}
    categories = reuters.categories()
    corpus = reuters.sents(categories=categories)
    for doc in corpus:
        # 增加起始字符
        doc = ['<s>'] + doc
        for i in range(0, len(doc) - 1):
            # bigram: [i,i+1]
            term = doc[i]
            bigram = doc[i:i + 2]

            if term in term_count:
                term_count[term] += 1
            else:
                term_count[term] = 1
            bigram = ' '.join(bigram)
            if bigram in bigram_count:
                bigram_count[bigram] += 1
            else:
                bigram_count[bigram] = 1
    return term_count, bigram_count


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


term_count, bigram_count = construct_lang_model()
channel_proba = channel_prob('spell-errors.txt')
V = len(term_count)
with open("testdata.txt", 'r', encoding="utf-8") as reader:
    lines = reader.readlines()
    for line in lines:
        items = line.rstrip().split("\t")
        words, corrected_words = spelling_correct(items[2], vocab, V)
        print("Sentence: %s" % items[2])
        corrected_sent = items[2]
        for word, corrected_word in zip(words, corrected_words):
            print("Corrected: %s => %s" % (word, corrected_word))
            corrected_sent = corrected_sent.replace(word, corrected_word)
        print("After: %s" % corrected_sent)