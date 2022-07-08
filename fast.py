from sklearn.model_selection import train_test_split
import fasttext

X, y = [], []
X_train, X_test, y_train, y_test = train_test_split(X, y)
with open('train.txt', 'w') as f:
    for target, text in zip(y_train, X_train):
        f.write(f'__label__{target} {text}\n')

with open('test.txt', 'w') as f:
    for target, text in zip(y_test, X_test):
        f.write(f'__label__{target} {text}\n')

model = fasttext.train_supervised('train.txt')
with open('test.txt', 'r') as f:
    X = [' '.join(line.split(' ')[1:]).rstrip() for line in f]
with open('test.txt', 'r') as f:
    true = [line.split(' ')[0] for line in f]
pred = [model.predict(text)[0][0] for text in X]
print(sum(t == p for t, p in zip(true, pred)) / len(true))
