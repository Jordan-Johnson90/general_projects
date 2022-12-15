# data analysis and wrangling
import polars as pl
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

train_df = pl.read_csv("/Users/Jordan/Documents/Python/Data/Titanic/original/train.csv")
test_df = pl.read_csv("/Users/Jordan/Documents/Python/Data/Titanic/original/test.csv")
combine = pl.concat([train_df, test_df], rechunk=True, how="diagonal")

combine.columns

for col in combine.get_columns():
    print(f"{col.name} - {col.is_null().sum()}")

train_df.describe()

train_df.select(["Name", "Sex", "Ticket", "Cabin", "Embarked"]).describe()

train_df.select(["Pclass", "Survived"]).groupby("Pclass").agg(
    pl.col("Survived").mean()
).sort("Survived", reverse=True)

train_df.select(["SibSp", "Survived"]).groupby("SibSp").agg(
    pl.col("Survived").mean()
).sort("Survived", reverse=True)

train_df.select(["Parch", "Survived"]).groupby("Parch").agg(
    pl.col("Survived").mean()
).sort("Survived", reverse=True)

g = sns.FacetGrid(data=train_df.to_pandas(), col="Survived")
g.map(plt.hist, "Age", bins=20)
plt.show()

grid = sns.FacetGrid(
    data=train_df.to_pandas(), col="Survived", row="Pclass", height=2.2, aspect=1.6
)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend()
plt.show()

grid = sns.FacetGrid(data=train_df.to_pandas(), row="Embarked", height=2.2, aspect=1.6)
grid.map(sns.pointplot, "Pclass", "Survived", "Sex", palette="deep")
grid.add_legend()
plt.show()

grid = sns.FacetGrid(
    data=train_df.to_pandas(), row="Embarked", col="Survived", height=2.2, aspect=1.6
)
grid.map(sns.barplot, "Sex", "Fare", alpha=0.5, ci=None)
grid.add_legend()
plt.show()

train_df.drop_in_place("Ticket")
train_df.drop_in_place("Cabin")
test_df.drop_in_place("Ticket")
test_df.drop_in_place("Cabin")
combine.drop_in_place("Ticket")
combine.drop_in_place("Cabin")

titles = (
    combine.with_column(pl.col("Name").str.extract(r" ([A-Za-z]+)\.", 1).alias("title"))
    .groupby(["title", "Sex"])
    .agg(pl.col("title").count().alias("count"))
    .pivot(values="count", index="title", columns="Sex")
)

rare_titles = [
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
]

combine = combine.with_column(
    pl.col("Name").str.extract(r" ([A-Za-z]+)\.", 1).alias("title")
)

survival_by_title = (
    combine.with_column(pl.col("Name").str.extract(r" ([A-Za-z]+)\.", 1).alias("title"))
    .with_column(
        pl.when(pl.col("title").is_in(rare_titles))
        .then("rare")
        .otherwise(pl.col("title"))
        .alias("title")
    )
    .with_column(pl.col("title").str.strip().str.replace("Mlle", "Miss").alias("title"))
    .with_column(pl.col("title").str.strip().str.replace("Ms", "Miss").alias("title"))
    .with_column(pl.col("title").str.strip().str.replace("Mme", "Mrs").alias("title"))
    .groupby(["title"])
    .agg(pl.col("Survived").mean().alias("survival_rate"))
    .sort("survival_rate", reverse=True)
)

title_mapping = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Rare": 5,
}  # is there a polars function like pandas' .map()?

combine = combine.with_column(
    pl.when(pl.col("title") == "Mr")
    .then("1")
    .when(pl.col("title") == "Miss")
    .then("2")
    .when(pl.col("title") == "Mrs")
    .then("3")
    .when(pl.col("title") == "Master")
    .then("4")
    .when(pl.col("title") == "Rare")
    .then("5")
    .otherwise("0")
    .alias("title")
)

train_df.drop_in_place("Name")
train_df.drop_in_place("PassengerId")
test_df.drop_in_place("Name")
test_df.drop_in_place("PassengerId")
combine.drop_in_place("Name")
combine.drop_in_place("PassengerId")

combine = combine.with_column(
    pl.when(pl.col("Sex") == "male")
    .then("0")
    .otherwise("1")
    .alias("Sex")
    .cast(pl.Int16)
)

grid = sns.FacetGrid(
    data=train_df.to_pandas(), row="Pclass", col="Sex", size=2.2, aspect=1.6
)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend()
plt.show()


guess_ages = np.zeros((2, 3))

# for i in range(0, 2):
#     for j in range(0, 3):
#         guess_df = (
#             combine.filter((pl.col("Sex") == i) & (pl.col("Pclass") == j + 1))
#             .select(pl.col("Age"))
#             .drop_nulls()
#         )

#         age_guess = guess_df.median()
#         guess_ages[i, j] = age_guess / 0.5 + 0.5


guess_ages = (
    combine.groupby(["Sex", "Pclass"])
    .agg(pl.col("Age").median())
    .with_column(((pl.col("Age") / 0.5 + 0.5) * 0.5).alias("age_guess"))
    .drop("Age")
)

combine = (
    combine.join(guess_ages, on=["Sex", "Pclass"], how="left")
    .with_column(
        pl.when(pl.col("Age").is_null())
        .then(pl.col("age_guess"))
        .otherwise(pl.col("Age"))
        .alias("Age")
    )
    .with_column(
        pl.when(pl.col("Age") <= 16)
        .then(0)
        .when(pl.col("Age") <= 32)
        .then(1)
        .when(pl.col("Age") <= 48)
        .then(2)
        .when(pl.col("Age") <= 64)
        .then(3)
        .otherwise(4)
        .keep_name()
    )
    .drop("age_guess")
    .with_column((pl.col("SibSp") + pl.col("Parch") + 1).alias("family_size"))
    .with_column(
        pl.when(pl.col("family_size") == 1).then(1).otherwise(0).alias("is_alone")
    )
)


survival_by_family_size = (
    combine.groupby("family_size")
    .agg(pl.col("Survived").mean())
    .sort("Survived", reverse=True)
)

is_alone_survival_rate = combine.groupby("is_alone").agg(pl.col("Survived").mean())

train_df.drop_in_place("Parch")
train_df.drop_in_place("SibSp")
test_df.drop_in_place("Parch")
test_df.drop_in_place("SibSp")
combine.drop_in_place("Parch")
combine.drop_in_place("SibSp")

combine = combine.with_column((pl.col("Age") * pl.col("Pclass")).alias("age*pclass"))

port_frequency = (combine.select("Embarked").drop_nulls()).to_series().mode()
combine = combine.with_column(pl.col("Embarked").fill_null(port_frequency))

survival_by_embarked = (
    combine.groupby("Embarked")
    .agg(pl.col("Survived").mean())
    .sort("Survived", reverse=True)
)

fare_mode = (
    combine.select(pl.col("Fare").cast(pl.Int16)).drop_nulls().to_series().mode()
)

combine = combine.with_columns(
    [
        pl.when(pl.col("Embarked") == "S")
        .then(0)
        .when(pl.col("Embarked") == "Q")
        .then(1)
        .when(pl.col("Embarked") == "C")
        .then(2)
        .otherwise(None)
        .alias("Embarked")
        .cast(pl.Int16),
        (pl.col("Fare").fill_null(fare_mode)).cast(pl.Float32),
    ]
).with_columns(
    [
        pl.when(pl.col("Fare") <= 7.91)
        .then(0)
        .when(pl.col("Fare") <= 14.454)
        .then(1)
        .when(pl.col("Fare") <= 31)
        .then(2)
        .otherwise(3)
        .cast(pl.Int16)
        .keep_name(),
        (pl.col("title").cast(pl.Int16)),
    ]
)

train = combine.filter(pl.col("Survived").is_not_null())
test = combine.filter(pl.col("Survived").is_null())
X_train = train.drop("Survived")
y_train = train.select("Survived")
X_test = test.clone().drop("Survived")
X_train.shape, y_train.shape, X_test.shape

# logistic regression
logreg = LogisticRegression()
logreg.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
y_pred = logreg.predict(X_test.to_numpy())
acc_log = round(logreg.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

# support vector machines
svc = SVC()
svc.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = svc.predict(X_test.to_numpy())
acc_svc = round(svc.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

# knn classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = knn.predict(X_test.to_pandas())
acc_knn = round(knn.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

# gaussian naive bayes
gaussian = GaussianNB()
gaussian.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = gaussian.predict(X_test.to_pandas())
acc_gaussian = round(gaussian.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

# perceptron
perceptron = Perceptron()
perceptron.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = perceptron.predict(X_test.to_pandas())
acc_perceptron = round(
    perceptron.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2
)

# linear svc
linear_svc = LinearSVC()
linear_svc.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = linear_svc.predict(X_test.to_pandas())
acc_linear_svc = round(
    linear_svc.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2
)

# stochastic gradient descent
sgd = SGDClassifier()
sgd.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = sgd.predict(X_test.to_pandas())
acc_sgd = round(sgd.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

# decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = decision_tree.predict(X_test.to_pandas())
acc_decision_tree = round(
    decision_tree.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2
)

# random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = random_forest.predict(X_test.to_pandas())
acc_random_forest = round(
    random_forest.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2
)

# xgboost
xgboost = XGBClassifier()
xgboost.fit(X_train.to_pandas(), y_train.to_numpy().ravel())
y_pred = xgboost.predict(X_test.to_pandas())
acc_xgboost = round(xgboost.score(X_train.to_pandas(), y_train.to_pandas()) * 100, 2)

data = {
    "Model": [
        "Support Vector Machines",
        "KNN",
        "Logistic Regression",
        "Random Forest",
        "Naive Bayes",
        "Perceptron",
        "Stochastic Gradient Decent",
        "Linear SVC",
        "Decision Tree",
        "XGBoost",
    ],
    "Score": [
        acc_svc,
        acc_knn,
        acc_log,
        acc_random_forest,
        acc_gaussian,
        acc_perceptron,
        acc_sgd,
        acc_linear_svc,
        acc_decision_tree,
        acc_xgboost,
    ],
}
models = pl.DataFrame(data).sort("Score", reverse=True)
