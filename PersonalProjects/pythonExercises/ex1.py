# Dataset : data = ["apple", "banana", "apple", "orange", "banana", "apple"]
# 👉 Tasks:

# Count frequency of each fruit
# Remove duplicates
# Sort the list

#check if running from virtual env
import sys
print(sys.executable)

data = ["apple", "banana", "apple", "orange", "banana", "apple"]
# Count frequency of each fruit
from collections import Counter
print(Counter(data))

fruitSet = set(data)

uniqueFruitCount = len(fruitSet)

print('uniqueFruitCount =',uniqueFruitCount)

#sort and no duplicates
noDuplicateList = list(fruitSet) 
noDuplicateList.sort()
print(noDuplicateList)


#if sorting without inbuilt function
def quicksort(lst):
    if not lst:
        return []
    return (quicksort([x for x in lst[1:] if x <  lst[0]])
            + [lst[0]] +
            quicksort([x for x in lst[1:] if x >= lst[0]]))

# 🏠 Homework

# Write a function:

# def analyze_text(text: str):
#     ...

# It should return:

# word count
# unique words
# most frequent word

# 👉 Input:

# "people like apple and people like banana"

inputStr = "people like apple and people like banana"

def analyze_text(text: str):
    word_array = text.split()
    wordCount = len(word_array)
    print(word_array)
    counts = Counter(word_array)
    most_frequent_word = max(counts, key=counts.get)
    word_set = set(word_array)
    uniqueWords = list(word_set) 
    uniqueWords.sort()
    print(uniqueWords)
    return wordCount, most_frequent_word, uniqueWords

print(analyze_text(inputStr))

data2 = ["apple", "banana", "apple"]
freq = Counter(data2)
print(freq)


print('# ----------------------Ideal solutions---------------------------')
# -------------------------------------------------
# Ideal solution 1
from collections import Counter

data = ["apple", "banana", "apple", "orange", "banana", "apple"]

# Frequency
freq = Counter(data)

# Unique + sorted
unique_sorted = sorted(set(data))

print(freq)
print(unique_sorted)
# -------------------------------------------------
# Ideal solution 2
from collections import Counter

def analyze_text(text: str):
    words = text.split()
    counts = Counter(words)

    return {
        "word_count": len(words),
        "unique_words": sorted(counts.keys()),
        "most_frequent": counts.most_common(1)[0][0]
    }

inputStr = "people like apple and people like banana"
print(analyze_text(inputStr))



