import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def Task01():
 # Wczytaj dane z plików (positive)
 with open('positive.txt', 'r', encoding='utf-8') as file:
  positive_data = file.readlines()

 # Wczytaj dane z plików (negative)
 with open('negative.txt', 'r', encoding='utf-8') as file:
  negative_data = file.readlines()

 # Przydziel klasy (0 - positive, 1 - negative)
 dataframe_1 = pd.DataFrame({'text': positive_data, 'class': 0})
 dataframe_2 = pd.DataFrame({'text': negative_data, 'class': 1})

 # Połącz ramki danych
 df = pd.concat([dataframe_1, dataframe_2], ignore_index=True)

 # Przemieszaj zbiór danych
 df = shuffle(df)

 # Przedstawienie za pomocą df.head()
 print(df.head())

 vectorizer = CountVectorizer()
 X = vectorizer.fit_transform(df['text'])
 X_positive = X[df['class'] == 0]
 X_negative = X[df['class'] == 1]

 # Sumowanie wystąpień każdego słowa w całym zbiorze oraz dla każdej klasy
 word_counts = X.toarray().sum(axis=0)
 word_counts_positive = X_positive.toarray().sum(axis=0)
 word_counts_negative = X_negative.toarray().sum(axis=0)

 # Pobranie słów odpowiadających tym wystąpieniom
 word_list = vectorizer.get_feature_names_out()

 # Połączenie słów z ich częstotliwościami w DataFrame oraz dla każdej klasy
 word_freq = pd.DataFrame({'word': word_list, 'count': word_counts})
 word_freq_positive = pd.DataFrame({'word': word_list, 'count': word_counts_positive})
 word_freq_negative = pd.DataFrame({'word': word_list, 'count': word_counts_negative})

 # Posortowanie według liczby wystąpień
 word_freq = word_freq.sort_values(by='count', ascending=False)
 word_freq_positive = word_freq_positive.sort_values(by='count', ascending=False)
 word_freq_negative = word_freq_negative.sort_values(by='count', ascending=False)

 # Wyświetlenie najczęstszych słów
 print("Najczęstsze słowa:")
 print(word_freq.head(10))

 print("Najczęstsze słowa dla zbioru positive:")
 print(word_freq_positive.head(10))

 print("\nNajczęstsze słowa dla zbioru negative:")
 print(word_freq_negative.head(10))

def Task02():
 # Wczytaj dane
 with open('positive.txt', 'r', encoding='utf-8') as file:
  positive_data = file.readlines()

 with open('negative.txt', 'r', encoding='utf-8') as file:
  negative_data = file.readlines()

 # Przydziel klasy (0 - positive, 1 - negative)
 positive_df = pd.DataFrame({'text': positive_data, 'class': 0})
 negative_df = pd.DataFrame({'text': negative_data, 'class': 1})

 # Połącz ramki danych
 df = pd.concat([positive_df, negative_df], ignore_index=True)

 # Przemieszaj zbiór danych
 df = shuffle(df)

 # Podziel dane na zbiór treningowy i testowy
 X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=42)

 classifiers = [
  ('Decision Tree', DecisionTreeClassifier()),
  ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
  ('SVM', SVC())
 ]

 vectorizers = [
  ('Default', CountVectorizer()),
  ('No Short Words (min 3 chars)', CountVectorizer(token_pattern=r'\b\w{3,}\b'))
 ]

 results = []

 for vectorizer_name, vectorizer in vectorizers:
  for classifier_name, classifier in classifiers:
   # Utwórz pipeline z wybranym vectorizer i klasyfikatorem
   pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
   cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

   # Trenuj model
   pipeline.fit(X_train, y_train)

   # Przewiduj na danych testowych
   y_pred = pipeline.predict(X_test)

   # Oceniaj wyniki
   accuracy = accuracy_score(y_test, y_pred)
   report = classification_report(y_test, y_pred)

   # Dodaj wyniki do listy
   results.append({
    'Vectorizer': vectorizer_name,
    'Classifier': classifier_name,
    'Mean Accuracy': cv_scores.mean(),
    'Cross-Validation Scores': cv_scores,
    'Classification Report': report
   })

 for result in results:
  print(f"Vectorizer: {result['Vectorizer']}")
  print(f"Classifier: {result['Classifier']}")
  print(f"Cross-Validation Scores: {result['Cross-Validation Scores']}")
  print(f"Mean CV Accuracy: {result['Mean Accuracy']:.4f}")
  print("Classification Report:")
  print(result['Classification Report'])
  print("=" * 50)

def main():
 Task02()

if __name__ == '__main__':
    main()