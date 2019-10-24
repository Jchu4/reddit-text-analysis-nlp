# Reddit Text Analysis and NLP

# Problem Statement
2 threads from reddit were chosen: 
- [r/askscience](https://www.reddit.com/r/askscience/): Using science to answer life's questions
- [r/askphilosophy](https://www.reddit.com/r/askphilosophy/): Using philosophical thinking to answer life's questions

Applied word normalisation & embedding techniques and determined a classification problem for a model to differentiate between the 2 threads


# Executive Summary
- Approximately 1,000+ posts were scraped from each of the chosen subreddits threads from reddit using the `BeautifulSoup`, `requests` & `json` libraries

- Data was then exported as *.csv* files, and then loaded into dataframes via `pandas`

- Word normalisation techniques such as  tokenizing, lemmatization were applied.

- Selected between 3 models, Multinomial Naive Bayes, KNN and Logistic Regression.For each model, 2 different word embedding methods were used (count vectorization and Tf-idf vectorization)

- For the modeling process, pipelines were used in conjunction with GridSearchCV to tune hyperparameters

- All models constructed performed well (except KNN), with the `MultinomialNB` model utilizing `TfidfVectorizer` as the word embedding method performing the best


- **Insights**:
    - Comparing the 3 different classification models: `MultinomialNB`, `Logistic Regression`, and `KNearestNeighbours` we can see that `MultinomialNB` produces the highest accuracy scores consistently acorss both vectorisation methods.

    - The selected model, which Tf-Idf vectorised and then classified using `MultinomialNB` was what gave us the best accuracy scores of 0.97 and 0.92 on the training and test datasets respectively.
        - Achieving a score of 0.92 would mean that we would be able to differentatiate between a *r/askscience* between a *r/askphilosophy* thread 9 out of 10 times correctly.
        - A true positive meant that the model correctly predicted a *r/askscience* post, while a true negative meant that the model correctly predicted a *r/askhilosophy* reddit post. 
            - This rate compared to the baseline accuracy was within a 89% - 95% range
           
- **Limitations**:
    - The model chosen `MultinomialNB`,  is only effective when used on smaller sample sizes, anything above 100k and we would like have chosen another model.
    - The `MultinomialNB` model makes a very strong assumption on the shape of your data distribution, i.e. any two features are independent given the output class. Hence, the result could have been potentially bad.
