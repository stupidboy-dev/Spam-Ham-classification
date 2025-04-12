```markdown
# Классификация спам-писем (Spam/Ham Email Classification)

Проект для автоматического определения спама в электронных письмах с использованием машинного обучения.

## Описание
Модель анализирует текст письма и классифицирует его как:
- **Ham** (0) — полезное письмо,
- **Spam** (1) — спам.

**Алгоритмы**: SVM (Support Vector Machine), TF-IDF для векторизации текста.

## Требования
- Данные: CSV-файл с колонками:
  - `email_text` (текст письма),
  - `label` (метка: `ham`/`spam`).

Пример данных:
```csv
email_text,label
"Win $1000 now! Click here...",spam
"Meeting at 3 PM tomorrow",ham
```

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone [ваш-репозиторий]
   ```
2. Установите зависимости:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost imbalanced-learn spacy
   ```
3. Загрузите ресурсы NLTK:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Использование
1. Загрузите данные:
   ```python
   data = pd.read_csv("spam_mail_classifier.csv")
   ```
2. Запустите обработку и обучение:
   ```python
   # Предобработка текста
   cleaned_texts = [clean_text(t) for t in data['email_text']]
   normalized_texts = [normalize_text(t) for t in cleaned_texts]
   
   # Векторизация
   vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
   X = vectorizer.fit_transform(normalized_texts)
   
   # Обучение модели
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   svm = SVC()
   svm.fit(X_train, y_train)
   
   # Оценка
   pred = svm.predict(X_test)
   print(classification_report(y_test, pred))
   ```

## Результаты
Пример метрик:
classification report:
```
          precision    recall  f1-score   support

           0       1.00      1.00      1.00       126
           1       1.00      1.00      1.00        74

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```
f1_score: 
``` 
1.0
```
confusion_matrix: 
```
[[126   0]
 [  0  74]]
```
