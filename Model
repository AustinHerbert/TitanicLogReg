{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "___\n",
    "# Logistic Regression Analysis\n",
    "\n",
    "I'll be trying to predict a classification- survival or deceased.\n",
    "First implementing Logistic Regression in Python for classification.\n",
    "\n",
    "\n",
    "\n",
    "## Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The Data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('titanic_train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis\n",
    "\n",
    "\n",
    "\n",
    "## Missing Data\n",
    "\n",
    "Used seaborn to create a simple heatmap to see where we are missing data!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Roughly 20% of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Who Survived?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,4))\n",
    "sns.set_style('whitegrid')\n",
    "sns.countplot(x='Survived',data=train,palette='RdBu_r',hue='Sex');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    549\n",
       "1    342\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Data show only 62% of passengers survived, with the majority of them being female."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Survived by Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,4))\n",
    "sns.set_style('whitegrid')\n",
    "sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Overwhelming the 3rd class passengers did not survive, whereas the survived passengers were more or less balanced."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### More Passenger Details"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "29.69911764705882"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(train['Age'].dropna(),bins=30)\n",
    "train['Age'].dropna().mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Average age was 29 with fairly normal distribution. Infants being a bit of an outlier on the curve."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Sibling/ Spouse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='SibSp',data=train,palette='rainbow');\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Most passengers were single or with just a spouse."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cufflinks for plots\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-latest.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import cufflinks as cf\n",
    "cf.go_offline()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    891.000000\n",
       "mean      32.204208\n",
       "std       49.693429\n",
       "min        0.000000\n",
       "25%        7.910400\n",
       "50%       14.454200\n",
       "75%       31.000000\n",
       "max      512.329200\n",
       "Name: Fare, dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Fare prices, not adjusted for inflation\n",
    "plt.figure(figsize=(12,4))\n",
    "sns.distplot(train['Fare'],kde=False,bins=40)\n",
    "train['Fare'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "## Data Cleaning\n",
    "To fill in missing age data instead of just dropping the missing age data rows. I gathered the mean age of all the passengers (imputation).\n",
    "\n",
    "- However we can be smarter about this and check the average age by passenger class. For example:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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o+gAAAClo+gAAAClo+gAAAClo+gAAAClo+gAAAClo+gAAAClo+gAAAClo+gAAAClo+gAAAClU7Wi8/DDD6ezszNJsmPHjlx99dXp6OjI4sWLMzQ0VK1hAQAAqlN0Pve5z2XBggUZGBhIkixZsiRz585Nd3d3KpVKNm7cWI1hAQAAklSp6EyaNCkrV64cPt66dWumTZuWJJkxY0a+/e1vV2NYAACAJElTNZ60vb09v/jFL4aPK5VKGhoakiRjx47N3r17j/kcAwMD6enpqUY86kh/f3+SeC8A5gNgmPmAkahK0fl9jY3/t3C0b9++jBs37pg/09zcnNbW1mrGog60tLQkifcCYD4AhpkPeKYjFd4TctW1V77yldmyZUuSZPPmzZk6deqJGBYAADhJnZCiM3/+/KxcuTJXXnllDh48mPb29hMxLAAAcJKq2qlrZ599du66664kybnnnps1a9ZUaygAAIBDuGEoAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOooOAABQOk1FBwCgOjZs2JB169YVHeO42b59e5Kkq6ur4CTHT3t7e9ra2oqOAVBKVnQAqAujR4/Ovn37snfv3qKjAFAHrOgAlFRbW1upVgtmzZqVJNm1a1dWrVpVcBoAap0VHQBq3ve+97309fUlSfr6+vL973+/4EQA1DpFB4Cat2TJkkOOP/rRjxaUBIB6oegAUPOeXs050jEA/D5FB4Cad9pppx31GAB+n6IDQM278cYbDzn+27/924KSAFAvFB0Aat4ZZ5xx1GMA+H2KDgA177bbbjvkeOnSpQUlAaBeKDoA1LydO3cecrxjx46CkgBQLxQdAGrepEmTDjk+55xzCkoCQL1QdACoeVdfffUhx7Nnzy4oCQD1oqnoABw/GzZsyLp164qOcVxt3749SdLV1VVwkuOnvb09bW1tRceAuvL5z3/+kON//Md/zIwZMwpKA0A9UHSoaePHjy86AlADfv3rXx9y/Ktf/aqgJADUC0WnRNra2qwUAABATmDRGRoays0335wf/ehHGTVqVG699VabSQEYkTFjxmT//v2HHAMjV7bT253azkic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\n",
      "text/plain": [
       "<Figure size 1008x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(14, 7))\n",
    "sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    False\n",
       "1    False\n",
       "2    False\n",
       "3    False\n",
       "4    False\n",
       "Name: Age, dtype: bool"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Age'].isnull().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    22.0\n",
       "1    38.0\n",
       "2    26.0\n",
       "3    35.0\n",
       "4    35.0\n",
       "Name: Age, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Age'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Replacing null values with avg age by specific class\n",
    "\n",
    "def impute_age(cols):\n",
    "    \n",
    "    #Index of cols you want to run function on\n",
    "    Age = cols[0]\n",
    "    Pclass = cols[1]\n",
    "    \n",
    "    #Missing Values from first col you pass in\n",
    "    if pd.isnull(Age):\n",
    "        \n",
    "        #Checks the class and returns value in missing value for Age\n",
    "        if Pclass == 1:\n",
    "            return 37\n",
    "\n",
    "        elif Pclass == 2:\n",
    "            return 29\n",
    "\n",
    "        else:\n",
    "            return 24\n",
    "\n",
    "    else:\n",
    "        return Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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4khNZvvwR3Z2tiZMmOBf3a86XeT2319ISIjRC0R/axMnTlR0dLTCw8MVFhamgwcPavPmzZo1a5bRusOGDdOyZct0880365ZbblGXLl1cfb5ruxfuvPNOLVy48Iy7w0yeShsxYoSWLl1qvGHziBEj/KvcDz74oPEOR+dbOfWpqXN3puXk5Jxx+OO7774zdvjD5t/frFmzXN8RUd0UFhZq48aNys/P11VXXaXevXsbP9ji89NPPykgIMD1eq6NdMPCwvxNWmyw0VDkbDZO4BCs7rN5+MPm319ND1yp8mbe/v37W625adMmzZgxQ40aNVJJSYlmzJih7t27u/Z810a6MTExKioq0o033uifczE5mX++/YImtrP06NFD4eHhchxHW7Zs8c8LSrzuAzXR4MGD9eqrryo4OFgFBQV69NFHXe174tpI1/QJrbOd70ptE9fZzJs3z/9zm/shAfw26tev7z/V17Rp0+q7Zezuu+/W2rVrdejQIXXv3l2tW7d269Hn5Wtk7DiOvvnmG2P7ZnndBy4Nvu135eXlGjt2rLp06aIdO3a4vjblWuhOmzbN6LXFZzt71GljJwOAmsu3/a7qNjwTB2hcC13T1xafrWoDk/z8fB06dMhoPQA1m2/7XWFhobZu3XrG9lc3uRa6pq8tPltCQoK/gUmTJk0uiZVcAOY9/PDDuv7669WwYUNJlacX3dzX7Vronn1tsaleAb7ba6s2MDl16pTxI5cALg0NGzZUUlKSsee7tmXM59ixY2rUqJHq1HEtz88QFRWliRMn6oYbblDfvn2VnJzsb2Biep8ugJpvyZIlqlev3hk3jXft2tW157uWjO+//74qKipUVlam5ORkjRo1SqNGjXLr8X4XamBiq68ugJrt888/V1lZmbZt2yapcnqhWobukiVLtGjRIsXGxuqjjz7Sww8/bCR0L9TApLi42PVaAC49JSUlRtvEuha6devWlVS5sdh3n5EJv8XttQAuHa1bt9a6devUrl07/+laN7u5uTan+9RTT2nr1q16+umntWvXLhUUFBi7tdNmAxMAlxbf1U6+aCwrK/PfKu0GVxfSiouLVb9+fR05csR/YgwA/hPExMT4j/2//vrr/unR4cOH+y+odYNrq0/btm3TF198oU2bNikyMlLvvvuuW48GAOOOHj3q//mmTZv8P3e7abproZucnKyWLVsqJSVFq1atYvsWgP9YJtu4uha6devW1eWXX646deqoadOmKisrc+vRAGBc1RGtySuBXNu90KBBAz300EO6//77tWLFijNukACA6u58Vy05jqOcnBxX67i2kFZWVqbc3Fxdf/31+v7779WyZUuj1/UAgJtsXbXkWuju27dPGRkZ/h4I+fn5xq8pB4D/NK7N6fq6fG3fvl0HDhzQ8ePH3Xo0ANQYroVuYGCgxo4dqyuuuELPP/+8jhw54tajAaDGcC10HcdRQUGBSkpKVFJSohMnTrj1aACoMVwJ3aKiIj322GPasGGD+vXrpz59+qhXr15uPBoAapSLXkhbvny5lixZojp16mjq1KmELQD8iose6a5bt04ZGRlKTU1VSkqKG18TANRYFx26Xq9XXq9XwcHBXJkDAP8LV69bMHleGQBqgoue0+3Ro4fCw8PlOI62bNniv81BkubMmXPRXyAA1CQXHbq2js4BQE3g+m3AAIAL4wpdALCI0AUAiwhdALCI0AUAiwhdALDovwEuOkC/DosI+gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Next I dropped the Cabin column, then single row in Embarked that is NaN."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.drop('Cabin',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Embarked  \n",
       "0      0         A/5 21171   7.2500        S  \n",
       "1      0          PC 17599  71.2833        C  \n",
       "2      0  STON/O2. 3101282   7.9250        S  \n",
       "3      0            113803  53.1000        S  \n",
       "4      0            373450   8.0500        S  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop all remaining NaN\n",
    "train.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Converting Categorical Features \n",
    "\n",
    "Convert categorical features to dummy variables otherwise our machine learning algorithm won't be able to directly take in those features as inputs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 889 entries, 0 to 890\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  889 non-null    int64  \n",
      " 1   Survived     889 non-null    int64  \n",
      " 2   Pclass       889 non-null    int64  \n",
      " 3   Name         889 non-null    object \n",
      " 4   Sex          889 non-null    object \n",
      " 5   Age          889 non-null    float64\n",
      " 6   SibSp        889 non-null    int64  \n",
      " 7   Parch        889 non-null    int64  \n",
      " 8   Ticket       889 non-null    object \n",
      " 9   Fare         889 non-null    float64\n",
      " 10  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(4)\n",
      "memory usage: 83.3+ KB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   2  3\n",
       "0  0  1"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get object cols into binary int\n",
    "sex = pd.get_dummies(train['Sex'],drop_first=True)\n",
    "embark = pd.get_dummies(train['Embarked'],drop_first=True)\n",
    "p_class = pd.get_dummies(train['Pclass'],drop_first=True)\n",
    "p_class.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop unused and old cols\n",
    "train.drop(['Sex','Embarked','Name','Ticket','Pclass','PassengerId'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.concat([train,sex,embark,p_class],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>male</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived   Age  SibSp  Parch     Fare  male  Q  S  2  3\n",
       "0         0  22.0      1      0   7.2500     1  0  1  0  1\n",
       "1         1  38.0      1      0  71.2833     0  0  0  0  0\n",
       "2         1  26.0      0      0   7.9250     0  0  1  0  1\n",
       "3         1  35.0      1      0  53.1000     0  0  1  0  0\n",
       "4         0  35.0      0      0   8.0500     1  0  1  0  1"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Building a Logistic Regression model\n",
    "\n",
    "\n",
    "## Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30, random_state=101)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training and Predicting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=1000,\n",
       "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel = LogisticRegression(max_iter=1000)\n",
    "logmodel.fit(X_train,y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = logmodel.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "from sklearn.feature_selection import SelectFromModel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.82      0.92      0.87       163\n",
      "           1       0.85      0.68      0.76       104\n",
      "\n",
      "    accuracy                           0.83       267\n",
      "   macro avg       0.83      0.80      0.81       267\n",
      "weighted avg       0.83      0.83      0.82       267\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,predictions))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
