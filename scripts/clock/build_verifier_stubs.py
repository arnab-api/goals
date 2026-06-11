"""Build the verifier-stub dataset for clock/05_verifier.

Each entry: a small self-contained coding task with a *correct* and an
*incorrect* implementation. The incorrect one is either a subtle SEMANTIC bug
(parses/compiles, runs, wrong output) or a BLATANT bug (syntax error / undefined
name / nonsense). We probe whether the code-LM, prefilled with the stub, closes
`</code>` and stops (verification pass) vs keeps generating alternates
(verification fail).

Run:  conda run -n goals python scripts/clock/build_verifier_stubs.py
Writes: data/verifier/stub_tasks.json
"""

import json
import os

# (lang, name, task, correct, incorrect, bug)   bug in {"semantic","blatant"}
T = []


def add(lang, name, task, correct, incorrect, bug):
    T.append(dict(lang=lang, name=name, task=task,
                  correct=correct.strip("\n"), incorrect=incorrect.strip("\n"), bug=bug))


# ============================== PYTHON — math / numbers ==============================
add("python", "factorial_rec", "Write a Python function `factorial(n)` that returns n! using recursion.",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
    "def factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result", "semantic")

add("python", "fib_iter", "Write a Python function `fib(n)` returning the n-th Fibonacci number (fib(0)=0, fib(1)=1).",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b", "semantic")

add("python", "is_prime", "Write a Python function `is_prime(n)` that returns True iff n is prime.",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return True\n    return False", "semantic")

add("python", "gcd", "Write a Python function `gcd(a, b)` returning the greatest common divisor.",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "def gcd(a, b):\n    return a % b", "semantic")

add("python", "lcm", "Write a Python function `lcm(a, b)` returning the least common multiple.",
    "def lcm(a, b):\n    x, y = a, b\n    while y:\n        x, y = y, x % y\n    return a * b // x",
    "def lcm(a, b):\n    return a * b", "semantic")

add("python", "is_even", "Write a Python function `is_even(n)` returning True iff n is even.",
    "def is_even(n):\n    return n % 2 == 0",
    "def is_even(n):\n    return n % 2 == 1", "semantic")

add("python", "sum_digits", "Write a Python function `sum_digits(n)` returning the sum of the decimal digits of n.",
    "def sum_digits(n):\n    n = abs(n)\n    total = 0\n    while n:\n        total += n % 10\n        n //= 10\n    return total",
    "def sum_digits(n):\n    return sum(str(n))", "blatant")

add("python", "reverse_int", "Write a Python function `reverse_int(n)` returning the digits of a non-negative n reversed.",
    "def reverse_int(n):\n    r = 0\n    while n:\n        r = r * 10 + n % 10\n        n //= 10\n    return r",
    "def reverse_int(n):\n    return n", "semantic")

add("python", "count_digits", "Write a Python function `count_digits(n)` returning the number of digits in non-negative n.",
    "def count_digits(n):\n    return len(str(n))",
    "def count_digits(n):\n    return n // 10", "semantic")

add("python", "is_power_of_two", "Write a Python function `is_power_of_two(n)` returning True iff n is a power of two.",
    "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0",
    "def is_power_of_two(n):\n    return n % 2 == 0", "semantic")

add("python", "sum_to_n", "Write a Python function `sum_to_n(n)` returning 1 + 2 + ... + n.",
    "def sum_to_n(n):\n    return n * (n + 1) // 2",
    "def sum_to_n(n):\n    return n * (n + 1)", "semantic")

add("python", "is_leap_year", "Write a Python function `is_leap_year(y)` returning True iff y is a leap year.",
    "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)",
    "def is_leap_year(y):\n    return y % 4 == 0", "semantic")

add("python", "celsius_to_f", "Write a Python function `c_to_f(c)` converting Celsius to Fahrenheit.",
    "def c_to_f(c):\n    return c * 9 / 5 + 32",
    "def c_to_f(c):\n    return c * 9 / 5", "semantic")

add("python", "abs_value", "Write a Python function `absval(x)` returning the absolute value of x without using abs().",
    "def absval(x):\n    return x if x >= 0 else -x",
    "def absval(x):\n    return -x", "semantic")

add("python", "clamp", "Write a Python function `clamp(x, lo, hi)` clamping x to the range [lo, hi].",
    "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))",
    "def clamp(x, lo, hi):\n    return x", "semantic")

add("python", "power", "Write a Python function `power(base, exp)` returning base**exp for non-negative integer exp.",
    "def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result",
    "def power(base, exp):\n    return base * exp", "semantic")

add("python", "is_perfect_square", "Write a Python function `is_perfect_square(n)` returning True iff n is a perfect square.",
    "def is_perfect_square(n):\n    if n < 0:\n        return False\n    r = int(n ** 0.5)\n    return r * r == n or (r + 1) * (r + 1) == n",
    "def is_perfect_square(n):\n    return n % 2 == 0", "semantic")

add("python", "nth_triangular", "Write a Python function `triangular(n)` returning the n-th triangular number.",
    "def triangular(n):\n    return n * (n + 1) // 2",
    "def triangular(n):\n    return sum(range(n))", "semantic")

add("python", "fib_rec", "Write a Python function `fib_rec(n)` returning the n-th Fibonacci number recursively.",
    "def fib_rec(n):\n    if n < 2:\n        return n\n    return fib_rec(n - 1) + fib_rec(n - 2)",
    "def fib_rec(n):\n    if n < 2:\n        return 1\n    return fib_rec(n - 1) + fib_rec(n - 2)", "semantic")

add("python", "factorial_inf", "Write a Python function `fact(n)` returning n! recursively.",
    "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)",
    "def fact(n):\n    return n * fact(n - 1)", "blatant")

# ============================== PYTHON — strings ==============================
add("python", "reverse_string", "Write a Python function `reverse(s)` returning the string s reversed.",
    "def reverse(s):\n    return s[::-1]",
    "def reverse(s):\n    return s", "semantic")

add("python", "is_palindrome", "Write a Python function `is_palindrome(s)` returning True iff s reads the same forwards and backwards.",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "def is_palindrome(s):\n    return s == s.upper()", "semantic")

add("python", "count_vowels", "Write a Python function `count_vowels(s)` returning the number of vowels in s.",
    "def count_vowels(s):\n    return sum(c in 'aeiouAEIOU' for c in s)",
    "def count_vowels(s):\n    return sum(c not in 'aeiouAEIOU' for c in s)", "semantic")

add("python", "count_words", "Write a Python function `count_words(s)` returning the number of whitespace-separated words in s.",
    "def count_words(s):\n    return len(s.split())",
    "def count_words(s):\n    return len(s)", "semantic")

add("python", "is_anagram", "Write a Python function `is_anagram(a, b)` returning True iff a and b are anagrams.",
    "def is_anagram(a, b):\n    return sorted(a) == sorted(b)",
    "def is_anagram(a, b):\n    return len(a) == len(b)", "semantic")

add("python", "capitalize_words", "Write a Python function `cap_words(s)` capitalizing the first letter of each word.",
    "def cap_words(s):\n    return ' '.join(w.capitalize() for w in s.split(' '))",
    "def cap_words(s):\n    return s.upper()", "semantic")

add("python", "count_char", "Write a Python function `count_char(s, ch)` returning how many times ch occurs in s.",
    "def count_char(s, ch):\n    return s.count(ch)",
    "def count_char(s, ch):\n    return ch in s", "semantic")

add("python", "swap_case", "Write a Python function `swap_case(s)` swapping the case of every letter.",
    "def swap_case(s):\n    return s.swapcase()",
    "def swap_case(s):\n    return s.lower()", "semantic")

add("python", "count_upper", "Write a Python function `count_upper(s)` returning the number of uppercase letters in s.",
    "def count_upper(s):\n    return sum(c.isupper() for c in s)",
    "def count_upper(s):\n    return sum(c.islower() for c in s)", "semantic")

add("python", "remove_spaces", "Write a Python function `remove_spaces(s)` returning s with all spaces removed.",
    "def remove_spaces(s):\n    return s.replace(' ', '')",
    "def remove_spaces(s):\n    return s.strip()", "semantic")

add("python", "first_char", "Write a Python function `first_char(s)` returning the first character of s, or '' if empty.",
    "def first_char(s):\n    return s[0] if s else ''",
    "def first_char(s):\n    return s[1]", "semantic")

add("python", "repeat_str", "Write a Python function `repeat(s, n)` returning s concatenated n times.",
    "def repeat(s, n):\n    return s * n",
    "def repeat(s, n):\n    return s + n", "blatant")

add("python", "starts_with", "Write a Python function `starts_with(s, p)` returning True iff s starts with prefix p.",
    "def starts_with(s, p):\n    return s.startswith(p)",
    "def starts_with(s, p):\n    return p in s", "semantic")

add("python", "reverse_str_bad", "Write a Python function `rev(s)` returning the string s reversed.",
    "def rev(s):\n    return ''.join(reversed(s))",
    "def rev(s):\n    return s.reverse()", "blatant")

add("python", "vowel_only_blatant", "Write a Python function `vowels(s)` returning a list of the vowels in s.",
    "def vowels(s):\n    return [c for c in s if c in 'aeiou']",
    "def vowels(s):\n    return [c for c in s if c in 'aeiou'", "blatant")

# ============================== PYTHON — lists ==============================
add("python", "sum_list", "Write a Python function `sum_list(lst)` returning the sum of the elements of lst.",
    "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total",
    "def sum_list(lst):\n    total = 1\n    for x in lst:\n        total += x\n    return total", "semantic")

add("python", "max_list", "Write a Python function `max_list(lst)` returning the largest element of a non-empty list.",
    "def max_list(lst):\n    m = lst[0]\n    for x in lst:\n        if x > m:\n            m = x\n    return m",
    "def max_list(lst):\n    return lst[0]", "semantic")

add("python", "min_list", "Write a Python function `min_list(lst)` returning the smallest element of a non-empty list.",
    "def min_list(lst):\n    m = lst[0]\n    for x in lst:\n        if x < m:\n            m = x\n    return m",
    "def min_list(lst):\n    m = 0\n    for x in lst:\n        if x < m:\n            m = x\n    return m", "semantic")

add("python", "average", "Write a Python function `average(lst)` returning the mean of a non-empty list.",
    "def average(lst):\n    return sum(lst) / len(lst)",
    "def average(lst):\n    return sum(lst) / (len(lst) - 1)", "semantic")

add("python", "count_even", "Write a Python function `count_even(lst)` returning how many elements of lst are even.",
    "def count_even(lst):\n    return sum(1 for x in lst if x % 2 == 0)",
    "def count_even(lst):\n    return sum(1 for x in lst if x % 2 == 1)", "semantic")

add("python", "filter_positive", "Write a Python function `positives(lst)` returning the positive elements of lst.",
    "def positives(lst):\n    return [x for x in lst if x > 0]",
    "def positives(lst):\n    return [x for x in lst if x < 0]", "semantic")

add("python", "reverse_list", "Write a Python function `reverse_list(lst)` returning a new list with the elements reversed.",
    "def reverse_list(lst):\n    return lst[::-1]",
    "def reverse_list(lst):\n    return lst", "semantic")

add("python", "contains", "Write a Python function `contains(lst, x)` returning True iff x is in lst.",
    "def contains(lst, x):\n    for y in lst:\n        if y == x:\n            return True\n    return False",
    "def contains(lst, x):\n    for y in lst:\n        if y == x:\n            return True\n        return False", "semantic")

add("python", "second_largest", "Write a Python function `second_largest(lst)` returning the second-largest distinct value.",
    "def second_largest(lst):\n    u = sorted(set(lst))\n    return u[-2]",
    "def second_largest(lst):\n    return sorted(lst)[-1]", "semantic")

add("python", "dot_product", "Write a Python function `dot(a, b)` returning the dot product of two equal-length lists.",
    "def dot(a, b):\n    return sum(x * y for x, y in zip(a, b))",
    "def dot(a, b):\n    return sum(a) + sum(b)", "semantic")

add("python", "running_sum", "Write a Python function `running_sum(lst)` returning the list of cumulative sums.",
    "def running_sum(lst):\n    out, s = [], 0\n    for x in lst:\n        s += x\n        out.append(s)\n    return out",
    "def running_sum(lst):\n    return lst", "semantic")

add("python", "flatten", "Write a Python function `flatten(nested)` flattening a list of lists by one level.",
    "def flatten(nested):\n    out = []\n    for sub in nested:\n        out.extend(sub)\n    return out",
    "def flatten(nested):\n    return nested", "semantic")

add("python", "remove_duplicates", "Write a Python function `dedup(lst)` returning lst with duplicates removed, preserving order.",
    "def dedup(lst):\n    seen, out = set(), []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
    "def dedup(lst):\n    return lst.unique()", "blatant")

add("python", "linear_search", "Write a Python function `find(lst, x)` returning the index of the first occurrence of x, or -1.",
    "def find(lst, x):\n    for i, y in enumerate(lst):\n        if y == x:\n            return i\n    return -1",
    "def find(lst, x):\n    for i, y in enumerate(lst):\n        if y == x:\n            return y\n    return -1", "semantic")

add("python", "index_of_max", "Write a Python function `argmax(lst)` returning the index of the largest element.",
    "def argmax(lst):\n    best = 0\n    for i in range(len(lst)):\n        if lst[i] > lst[best]:\n            best = i\n    return best",
    "def argmax(lst):\n    return lst.index(min(lst))", "semantic")

add("python", "product_list", "Write a Python function `product(lst)` returning the product of all elements.",
    "def product(lst):\n    p = 1\n    for x in lst:\n        p *= x\n    return p",
    "def product(lst):\n    p = 0\n    for x in lst:\n        p *= x\n    return p", "semantic")

add("python", "count_positive_blatant", "Write a Python function `n_pos(lst)` returning how many elements are positive.",
    "def n_pos(lst):\n    return sum(1 for x in lst if x > 0)",
    "def n_pos(lst):\n    return sum(1 for x in lst if x > 0", "blatant")

add("python", "list_sum_undefined", "Write a Python function `total(lst)` returning the sum of lst.",
    "def total(lst):\n    return sum(lst)",
    "def total(lst):\n    return summation(lst)", "blatant")

# ============================== PYTHON — search / sort ==============================
add("python", "binary_search", "Write a Python function `bsearch(lst, x)` returning the index of x in a sorted list, or -1.",
    "def bsearch(lst, x):\n    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == x:\n            return mid\n        if lst[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "def bsearch(lst, x):\n    lo, hi = 0, len(lst) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == x:\n            return mid\n        if lst[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1", "semantic")

add("python", "is_sorted", "Write a Python function `is_sorted(lst)` returning True iff lst is non-decreasing.",
    "def is_sorted(lst):\n    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))",
    "def is_sorted(lst):\n    return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))", "semantic")

add("python", "bubble_sort", "Write a Python function `bubble_sort(lst)` sorting a list ascending in place and returning it.",
    "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(n - 1 - i):\n            if lst[j] > lst[j + 1]:\n                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n    return lst",
    "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(n - 1):\n            if lst[j] > lst[j + 1]:\n                lst[j], lst[j + 1] = lst[j], lst[j]\n    return lst", "semantic")

add("python", "merge_sorted", "Write a Python function `merge(a, b)` merging two sorted lists into one sorted list.",
    "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
    "def merge(a, b):\n    return a + b", "semantic")

add("python", "selection_min", "Write a Python function `find_min(lst)` returning the minimum element of a non-empty list.",
    "def find_min(lst):\n    m = lst[0]\n    for x in lst[1:]:\n        if x < m:\n            m = x\n    return m",
    "def find_min(lst):\n    return lst[-1]", "semantic")

# ============================== PYTHON — misc ==============================
add("python", "fizzbuzz", "Write a Python function `fizzbuzz(n)` returning 'FizzBuzz'/'Fizz'/'Buzz'/str(n) for n.",
    "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
    "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    return str(n)", "semantic")

add("python", "grade", "Write a Python function `grade(score)` returning 'A' for >=90, 'B' for >=80, else 'F'.",
    "def grade(score):\n    if score >= 90:\n        return 'A'\n    if score >= 80:\n        return 'B'\n    return 'F'",
    "def grade(score):\n    if score >= 80:\n        return 'B'\n    if score >= 90:\n        return 'A'\n    return 'F'", "semantic")

add("python", "max_of_three", "Write a Python function `max3(a, b, c)` returning the largest of three numbers.",
    "def max3(a, b, c):\n    return max(a, b, c)",
    "def max3(a, b, c):\n    return a if a > b else b", "semantic")

add("python", "sign", "Write a Python function `sign(x)` returning -1, 0, or 1 for the sign of x.",
    "def sign(x):\n    return (x > 0) - (x < 0)",
    "def sign(x):\n    return 1 if x > 0 else -1", "semantic")

add("python", "celsius_round_blatant", "Write a Python function `kelvin(c)` converting Celsius to Kelvin.",
    "def kelvin(c):\n    return c + 273.15",
    "def kelvin(c):\n    return c + 273.15 +", "blatant")

add("python", "double_blatant", "Write a Python function `double(x)` returning x doubled.",
    "def double(x):\n    return x * 2",
    "def double(x):\n    retrun x * 2", "blatant")

add("python", "to_int_blatant", "Write a Python function `to_int(s)` parsing a decimal string to an int.",
    "def to_int(s):\n    return int(s)",
    "def to_int(s):\n    return int(s", "blatant")

add("python", "min_max_undefined", "Write a Python function `spread(lst)` returning max(lst) - min(lst).",
    "def spread(lst):\n    return max(lst) - min(lst)",
    "def spread(lst):\n    return biggest(lst) - smallest(lst)", "blatant")

add("python", "square_blatant", "Write a Python function `square(x)` returning x squared.",
    "def square(x):\n    return x * x",
    "def square(x):\n    return x ^ 2", "semantic")

add("python", "mean_blatant", "Write a Python function `mean(lst)` returning the average of lst.",
    "def mean(lst):\n    return sum(lst) / len(lst)",
    "def mean(lst):\n    return sum(lst) // len()", "blatant")

# ============================== C++ ==============================
add("cpp", "isPalindrome", "Write a C++ function `bool isPalindrome(string s)` returning true iff s is a palindrome.",
    "bool isPalindrome(string s){\n    int l = 0, r = s.size() - 1;\n    while (l < r) {\n        if (s[l] != s[r]) return false;\n        l++; r--;\n    }\n    return true;\n}",
    "bool isPalindrome(string s){\n    return s == s[::-1];\n}", "blatant")

add("cpp", "factorial", "Write a C++ function `long factorial(int n)` returning n!.",
    "long factorial(int n){\n    long r = 1;\n    for (int i = 2; i <= n; i++) r *= i;\n    return r;\n}",
    "long factorial(int n){\n    long r = 0;\n    for (int i = 2; i <= n; i++) r *= i;\n    return r;\n}", "semantic")

add("cpp", "isPrime", "Write a C++ function `bool isPrime(int n)` returning true iff n is prime.",
    "bool isPrime(int n){\n    if (n < 2) return false;\n    for (int i = 2; i * i <= n; i++)\n        if (n % i == 0) return false;\n    return true;\n}",
    "bool isPrime(int n){\n    if (n < 2) return false;\n    for (int i = 2; i < n; i++)\n        if (n % i == 0) return true;\n    return false;\n}", "semantic")

add("cpp", "gcd", "Write a C++ function `int gcd(int a, int b)` returning the greatest common divisor.",
    "int gcd(int a, int b){\n    while (b) { int t = b; b = a % b; a = t; }\n    return a;\n}",
    "int gcd(int a, int b){\n    return a % b;\n}", "semantic")

add("cpp", "isEven", "Write a C++ function `bool isEven(int n)` returning true iff n is even.",
    "bool isEven(int n){\n    return n % 2 == 0;\n}",
    "bool isEven(int n){\n    return n % 2 == 1;\n}", "semantic")

add("cpp", "sumArray", "Write a C++ function `int sumArray(vector<int> a)` returning the sum of the elements.",
    "int sumArray(vector<int> a){\n    int s = 0;\n    for (int x : a) s += x;\n    return s;\n}",
    "int sumArray(vector<int> a){\n    int s = 1;\n    for (int x : a) s += x;\n    return s;\n}", "semantic")

add("cpp", "maxArray", "Write a C++ function `int maxArray(vector<int> a)` returning the largest element of a non-empty vector.",
    "int maxArray(vector<int> a){\n    int m = a[0];\n    for (int x : a) if (x > m) m = x;\n    return m;\n}",
    "int maxArray(vector<int> a){\n    return a[0];\n}", "semantic")

add("cpp", "fibonacci", "Write a C++ function `int fib(int n)` returning the n-th Fibonacci number (fib(0)=0).",
    "int fib(int n){\n    int a = 0, b = 1;\n    for (int i = 0; i < n; i++) { int t = a + b; a = b; b = t; }\n    return a;\n}",
    "int fib(int n){\n    int a = 0, b = 1;\n    for (int i = 0; i < n; i++) { int t = a + b; a = b; b = t; }\n    return b;\n}", "semantic")

add("cpp", "reverseString", "Write a C++ function `string reverseString(string s)` returning s reversed.",
    "string reverseString(string s){\n    int l = 0, r = s.size() - 1;\n    while (l < r) { swap(s[l], s[r]); l++; r--; }\n    return s;\n}",
    "string reverseString(string s){\n    return s;\n}", "semantic")

add("cpp", "countVowels", "Write a C++ function `int countVowels(string s)` returning the number of vowels in s.",
    "int countVowels(string s){\n    int c = 0;\n    for (char ch : s)\n        if (string(\"aeiouAEIOU\").find(ch) != string::npos) c++;\n    return c;\n}",
    "int countVowels(string s){\n    int c = 0;\n    for (char ch : s)\n        if (string(\"aeiouAEIOU\").find(ch) == string::npos) c++;\n    return c;\n}", "semantic")

add("cpp", "power", "Write a C++ function `long power(int base, int exp)` returning base^exp for exp>=0.",
    "long power(int base, int exp){\n    long r = 1;\n    for (int i = 0; i < exp; i++) r *= base;\n    return r;\n}",
    "long power(int base, int exp){\n    return base * exp;\n}", "semantic")

add("cpp", "isLeapYear", "Write a C++ function `bool isLeapYear(int y)` returning true iff y is a leap year.",
    "bool isLeapYear(int y){\n    return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);\n}",
    "bool isLeapYear(int y){\n    return y % 4 == 0;\n}", "semantic")

add("cpp", "sumDigits", "Write a C++ function `int sumDigits(int n)` returning the sum of the digits of n>=0.",
    "int sumDigits(int n){\n    int s = 0;\n    while (n) { s += n % 10; n /= 10; }\n    return s;\n}",
    "int sumDigits(int n){\n    int s = 0;\n    while (n) { s += n % 10; }\n    return s;\n}", "semantic")

add("cpp", "absVal", "Write a C++ function `int absVal(int x)` returning the absolute value of x.",
    "int absVal(int x){\n    return x < 0 ? -x : x;\n}",
    "int absVal(int x){\n    return -x;\n}", "semantic")

add("cpp", "binarySearch", "Write a C++ function `int binarySearch(vector<int> a, int x)` returning the index of x or -1.",
    "int binarySearch(vector<int> a, int x){\n    int lo = 0, hi = a.size() - 1;\n    while (lo <= hi) {\n        int mid = (lo + hi) / 2;\n        if (a[mid] == x) return mid;\n        if (a[mid] < x) lo = mid + 1; else hi = mid - 1;\n    }\n    return -1;\n}",
    "int binarySearch(vector<int> a, int x){\n    int lo = 0, hi = length - 1;\n    while (lo <= hi) {\n        int mid = (lo + hi) / 2;\n        if (a[mid] == x) return mid;\n        if (a[mid] < x) lo = mid + 1; else hi = mid - 1;\n    }\n    return -1;\n}", "blatant")

add("cpp", "square_missing", "Write a C++ function `int square(int x)` returning x squared.",
    "int square(int x){\n    return x * x;\n}",
    "int square(int x){\n    return x * x\n}", "blatant")

add("cpp", "doubleVal_blatant", "Write a C++ function `int doubleVal(int x)` returning x doubled.",
    "int doubleVal(int x){\n    return x * 2;\n}",
    "int doubleVal(int x){\n    retrun x * 2;\n}", "blatant")

add("cpp", "minArray", "Write a C++ function `int minArray(vector<int> a)` returning the smallest element of a non-empty vector.",
    "int minArray(vector<int> a){\n    int m = a[0];\n    for (int x : a) if (x < m) m = x;\n    return m;\n}",
    "int minArray(vector<int> a){\n    int m = 0;\n    for (int x : a) if (x < m) m = x;\n    return m;\n}", "semantic")


# ============================== extra blatant (syntax / undefined / typo) ==============================
add("python", "cube_blatant", "Write a Python function `cube(x)` returning x cubed.",
    "def cube(x):\n    return x ** 3",
    "def cube(x):\n    return x ** ", "blatant")

add("python", "halve_blatant", "Write a Python function `halve(x)` returning x divided by two.",
    "def halve(x):\n    return x / 2",
    "def halve(x):\n    return x /", "blatant")

add("python", "negate_typo", "Write a Python function `negate(x)` returning -x.",
    "def negate(x):\n    return -x",
    "def negate(x):\n    reutrn -x", "blatant")

add("python", "join_blatant", "Write a Python function `join_words(lst)` joining a list of strings with spaces.",
    "def join_words(lst):\n    return ' '.join(lst)",
    "def join_words(lst):\n    return ' '.join(lst", "blatant")

add("python", "len_undefined", "Write a Python function `length(s)` returning the length of s.",
    "def length(s):\n    return len(s)",
    "def length(s):\n    return size(s)", "blatant")

add("python", "incr_typo", "Write a Python function `incr(x)` returning x + 1.",
    "def incr(x):\n    return x + 1",
    "def incr(x):\n    return x ++ 1", "blatant")

add("python", "last_elem_blatant", "Write a Python function `last(lst)` returning the last element of lst.",
    "def last(lst):\n    return lst[-1]",
    "def last(lst):\n    return lst[len(lst)]", "semantic")

add("cpp", "addOne_typo", "Write a C++ function `int addOne(int x)` returning x + 1.",
    "int addOne(int x){\n    return x + 1;\n}",
    "int addOne(int x){\n    return x + 1\n}", "blatant")

add("cpp", "negate_undefined", "Write a C++ function `int negate(int x)` returning -x.",
    "int negate(int x){\n    return -x;\n}",
    "int negate(int x){\n    return neg(x);\n}", "blatant")

add("cpp", "triple_blatant", "Write a C++ function `int triple(int x)` returning 3*x.",
    "int triple(int x){\n    return 3 * x;\n}",
    "int triple(int x){\n    return 3 * x +;\n}", "blatant")

add("cpp", "isOdd_typo", "Write a C++ function `bool isOdd(int n)` returning true iff n is odd.",
    "bool isOdd(int n){\n    return n % 2 != 0;\n}",
    "bool isOdd(int n){\n    retrun n % 2 != 0;\n}", "blatant")

add("cpp", "lenStr_undefined", "Write a C++ function `int lenStr(string s)` returning the length of s.",
    "int lenStr(string s){\n    return s.size();\n}",
    "int lenStr(string s){\n    return length(s);\n}", "blatant")

add("cpp", "sumTwo_blatant", "Write a C++ function `int sumTwo(int a, int b)` returning a + b.",
    "int sumTwo(int a, int b){\n    return a + b;\n}",
    "int sumTwo(int a, int b){\n    return a + b\n}", "blatant")

add("python", "cap_first_blatant", "Write a Python function `cap_first(s)` capitalizing the first letter of s.",
    "def cap_first(s):\n    return s[:1].upper() + s[1:]",
    "def cap_first(s):\n    return s[:1].upper( + s[1:]", "blatant")


# ============================================================================
# Curated "complete + blatant" set: the stub is a COMPLETE function that PARSES
# but is clearly wrong — either obviously-wrong logic (`wrong_logic`) or a call
# to a method/function that doesn't exist in this language though it is idiomatic
# in another (`wrong_lib`). No syntax typos / truncations here.
# ============================================================================
B = []


def addb(lang, name, task, correct, incorrect, kind):
    B.append(dict(lang=lang, name=name, task=task,
                  correct=correct.strip("\n"), incorrect=incorrect.strip("\n"),
                  bug="blatant", blatant_kind=kind))


# --- python: wrong-language / wrong-library idiom (parses, runtime-wrong) ---
addb("python", "reverse_string", "Write a Python function `reverse(s)` returning the string s reversed.",
     "def reverse(s):\n    return s[::-1]",
     "def reverse(s):\n    return s.reverse()", "wrong_lib")

addb("python", "to_upper", "Write a Python function `shout(s)` returning s in all uppercase.",
     "def shout(s):\n    return s.upper()",
     "def shout(s):\n    return s.toUpperCase()", "wrong_lib")

addb("python", "contains_substr", "Write a Python function `contains(s, sub)` returning True iff sub occurs in s.",
     "def contains(s, sub):\n    return sub in s",
     "def contains(s, sub):\n    return s.includes(sub)", "wrong_lib")

addb("python", "is_empty", "Write a Python function `is_empty(lst)` returning True iff lst has no elements.",
     "def is_empty(lst):\n    return len(lst) == 0",
     "def is_empty(lst):\n    return lst.isEmpty()", "wrong_lib")

addb("python", "dedup", "Write a Python function `dedup(lst)` returning the distinct elements of lst.",
     "def dedup(lst):\n    out = []\n    for x in lst:\n        if x not in out:\n            out.append(x)\n    return out",
     "def dedup(lst):\n    return lst.unique()", "wrong_lib")

addb("python", "abs_value", "Write a Python function `absval(x)` returning the absolute value of x.",
     "def absval(x):\n    return x if x >= 0 else -x",
     "def absval(x):\n    return Math.abs(x)", "wrong_lib")

addb("python", "str_length", "Write a Python function `length(s)` returning the number of characters in s.",
     "def length(s):\n    return len(s)",
     "def length(s):\n    return s.length()", "wrong_lib")

addb("python", "parse_int", "Write a Python function `to_int(s)` parsing a decimal string into an int.",
     "def to_int(s):\n    return int(s)",
     "def to_int(s):\n    return parseInt(s)", "wrong_lib")

addb("python", "sort_list", "Write a Python function `sort_list(lst)` returning a new ascending-sorted list.",
     "def sort_list(lst):\n    return sorted(lst)",
     "def sort_list(lst):\n    return lst.sort()", "wrong_lib")

addb("python", "first_elem", "Write a Python function `first(lst)` returning the first element of lst.",
     "def first(lst):\n    return lst[0]",
     "def first(lst):\n    return lst.first()", "wrong_lib")

# --- python: clearly-wrong logic (parses, obviously wrong) ---
addb("python", "factorial", "Write a Python function `factorial(n)` returning n!.",
     "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
     "def factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result", "wrong_logic")

addb("python", "is_even", "Write a Python function `is_even(n)` returning True iff n is even.",
     "def is_even(n):\n    return n % 2 == 0",
     "def is_even(n):\n    return n % 2 == 1", "wrong_logic")

# --- cpp: python/other idiom in C++ (complete, won't compile) ---
addb("cpp", "isPalindrome", "Write a C++ function `bool isPalindrome(string s)` returning true iff s is a palindrome.",
     "bool isPalindrome(string s){\n    int l = 0, r = s.size() - 1;\n    while (l < r) {\n        if (s[l] != s[r]) return false;\n        l++; r--;\n    }\n    return true;\n}",
     "bool isPalindrome(string s){\n    return s == s[::-1];\n}", "wrong_lib")

addb("cpp", "lenStr", "Write a C++ function `int lenStr(string s)` returning the length of s.",
     "int lenStr(string s){\n    return s.size();\n}",
     "int lenStr(string s){\n    return len(s);\n}", "wrong_lib")

addb("cpp", "reverseString", "Write a C++ function `string reverseString(string s)` returning s reversed.",
     "string reverseString(string s){\n    int l = 0, r = s.size() - 1;\n    while (l < r) { swap(s[l], s[r]); l++; r--; }\n    return s;\n}",
     "string reverseString(string s){\n    return s.reverse();\n}", "wrong_lib")


# ============================================================================
# Curated "wrong task" set: the stub is a COMPLETE function that PARSES and RUNS
# without error, but computes something clearly unrelated to what was asked
# (e.g. "is this a palindrome?" -> checks whether the first char is lowercase).
# No syntax errors, no missing library — only the semantics are off-task.
# ============================================================================
W = []


def addw(lang, name, task, correct, incorrect):
    W.append(dict(lang=lang, name=name, task=task,
                  correct=correct.strip("\n"), incorrect=incorrect.strip("\n"),
                  bug="blatant", blatant_kind="wrong_task"))


addw("python", "is_palindrome", "Write a Python function `is_palindrome(s)` returning True iff s reads the same forwards and backwards.",
     "def is_palindrome(s):\n    return s == s[::-1]",
     "def is_palindrome(s):\n    return s[0].islower()")

addw("python", "factorial", "Write a Python function `factorial(n)` returning n!.",
     "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
     "def factorial(n):\n    return n + 1")

addw("python", "is_prime", "Write a Python function `is_prime(n)` returning True iff n is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
     "def is_prime(n):\n    return n % 2 == 0")

addw("python", "reverse_string", "Write a Python function `reverse(s)` returning the string s reversed.",
     "def reverse(s):\n    return s[::-1]",
     "def reverse(s):\n    return s.upper()")

addw("python", "sum_list", "Write a Python function `sum_list(lst)` returning the sum of the elements of lst.",
     "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total",
     "def sum_list(lst):\n    return len(lst)")

addw("python", "count_vowels", "Write a Python function `count_vowels(s)` returning the number of vowels in s.",
     "def count_vowels(s):\n    return sum(c in 'aeiouAEIOU' for c in s)",
     "def count_vowels(s):\n    return s.count(' ')")

addw("python", "max_list", "Write a Python function `max_list(lst)` returning the largest element of a non-empty list.",
     "def max_list(lst):\n    m = lst[0]\n    for x in lst:\n        if x > m:\n            m = x\n    return m",
     "def max_list(lst):\n    return sum(lst)")

addw("python", "gcd", "Write a Python function `gcd(a, b)` returning the greatest common divisor of a and b.",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     "def gcd(a, b):\n    return a + b")

addw("python", "is_even", "Write a Python function `is_even(n)` returning True iff n is even.",
     "def is_even(n):\n    return n % 2 == 0",
     "def is_even(n):\n    return n > 0")

addw("python", "average", "Write a Python function `average(lst)` returning the mean of a non-empty list.",
     "def average(lst):\n    return sum(lst) / len(lst)",
     "def average(lst):\n    return lst[0]")

addw("python", "count_words", "Write a Python function `count_words(s)` returning the number of whitespace-separated words in s.",
     "def count_words(s):\n    return len(s.split())",
     "def count_words(s):\n    return s[::-1]")

addw("python", "fibonacci", "Write a Python function `fib(n)` returning the n-th Fibonacci number (fib(0)=0, fib(1)=1).",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     "def fib(n):\n    return n * 2")


# ============================================================================
# Curated "incomplete" set: the stub is on-task and SYNTACTICALLY COMPLETE
# (parses, runs, ends at a clean line — never cut off mid-line) but does not
# finish the task. Two sub-kinds:
#   missing_condition — checks/handles only some of the required cases
#   partial_steps     — a multi-step task that stops after the first step(s)
# ============================================================================
I = []


def addi(name, task, correct, incorrect, kind):
    I.append(dict(lang="python", name=name, task=task,
                  correct=correct.strip("\n"), incorrect=incorrect.strip("\n"),
                  bug="incomplete", sub_kind=kind))


# --- missing_condition: only some of the required cases are handled ---
addi("is_prime", "Write a Python function `is_prime(n)` returning True iff n is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
     "def is_prime(n):\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return False\n    if n % 3 == 0:\n        return False\n    return True", "missing_condition")

addi("fizzbuzz", "Write a Python function `fizzbuzz(n)` returning 'FizzBuzz' if n is divisible by 15, else 'Fizz' for 3, 'Buzz' for 5, else str(n).",
     "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
     "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)", "missing_condition")

addi("is_valid_password", "Write a Python function `is_valid(s)` returning True iff s has length >= 8, contains a digit, and contains an uppercase letter.",
     "def is_valid(s):\n    return len(s) >= 8 and any(c.isdigit() for c in s) and any(c.isupper() for c in s)",
     "def is_valid(s):\n    return len(s) >= 8", "missing_condition")

addi("is_leap_year", "Write a Python function `is_leap_year(y)` returning True iff y is a leap year (divisible by 4, except centuries not divisible by 400).",
     "def is_leap_year(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)",
     "def is_leap_year(y):\n    return y % 4 == 0", "missing_condition")

addi("valid_triangle", "Write a Python function `valid_triangle(a, b, c)` returning True iff three side lengths can form a triangle.",
     "def valid_triangle(a, b, c):\n    return a + b > c and a + c > b and b + c > a",
     "def valid_triangle(a, b, c):\n    return a + b > c", "missing_condition")

addi("count_vowels", "Write a Python function `count_vowels(s)` returning the number of vowels (a, e, i, o, u) in s.",
     "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())",
     "def count_vowels(s):\n    return sum(c in 'ae' for c in s.lower())", "missing_condition")

# --- partial_steps: a multi-step task that stops after the first step(s) ---
addi("normalize_name", "Write a Python function `normalize_name(s)` that strips surrounding whitespace, lowercases, then capitalizes the first letter.",
     "def normalize_name(s):\n    s = s.strip()\n    s = s.lower()\n    return s.capitalize()",
     "def normalize_name(s):\n    s = s.strip()\n    return s", "partial_steps")

addi("reverse_words", "Write a Python function `reverse_words(s)` returning the words of s in reverse order, space-joined.",
     "def reverse_words(s):\n    words = s.split()\n    words = words[::-1]\n    return ' '.join(words)",
     "def reverse_words(s):\n    words = s.split()\n    return ' '.join(words)", "partial_steps")

addi("sum_sq_evens", "Write a Python function `sum_sq_evens(lst)` returning the sum of the squares of the even numbers in lst.",
     "def sum_sq_evens(lst):\n    evens = [x for x in lst if x % 2 == 0]\n    squares = [x * x for x in evens]\n    return sum(squares)",
     "def sum_sq_evens(lst):\n    evens = [x for x in lst if x % 2 == 0]\n    return sum(evens)", "partial_steps")

addi("initials", "Write a Python function `initials(name)` returning the uppercase initials of every word in name (e.g. 'john f kennedy' -> 'JFK').",
     "def initials(name):\n    words = name.split()\n    letters = [w[0] for w in words]\n    return ''.join(letters).upper()",
     "def initials(name):\n    words = name.split()\n    return words[0][0].upper()", "partial_steps")

addi("count_long_words", "Write a Python function `count_long_words(s, n)` returning how many words in s have more than n characters.",
     "def count_long_words(s, n):\n    words = s.split()\n    long_words = [w for w in words if len(w) > n]\n    return len(long_words)",
     "def count_long_words(s, n):\n    words = s.split()\n    return len(words)", "partial_steps")

addi("clean_and_dedup", "Write a Python function `clean_and_dedup(lst)` that drops falsy elements and then removes duplicates, preserving order.",
     "def clean_and_dedup(lst):\n    kept = [x for x in lst if x]\n    out = []\n    for x in kept:\n        if x not in out:\n            out.append(x)\n    return out",
     "def clean_and_dedup(lst):\n    kept = [x for x in lst if x]\n    return kept", "partial_steps")


def _write(items, path):
    for i, e in enumerate(items):
        e["id"] = f"{e['lang']}_{i:03d}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    out_dir = os.path.join("data", "verifier")
    _write(T, os.path.join(out_dir, "stub_tasks.json"))
    n_sem = sum(e["bug"] == "semantic" for e in T)
    n_bla = sum(e["bug"] == "blatant" for e in T)
    print(f"wrote {len(T)} tasks -> {out_dir}/stub_tasks.json")
    print(f"  python={sum(e['lang']=='python' for e in T)}  cpp={sum(e['lang']=='cpp' for e in T)}"
          f"   semantic={n_sem}  blatant={n_bla} ({100 * n_bla / len(T):.0f}% blatant)")

    _write(B, os.path.join(out_dir, "blatant_complete.json"))
    print(f"wrote {len(B)} tasks -> {out_dir}/blatant_complete.json")
    print(f"  python={sum(e['lang']=='python' for e in B)}  cpp={sum(e['lang']=='cpp' for e in B)}"
          f"   wrong_lib={sum(e['blatant_kind']=='wrong_lib' for e in B)}"
          f"  wrong_logic={sum(e['blatant_kind']=='wrong_logic' for e in B)}")

    _write(W, os.path.join(out_dir, "wrong_task.json"))
    print(f"wrote {len(W)} tasks -> {out_dir}/wrong_task.json  (all wrong_task, "
          f"python={sum(e['lang']=='python' for e in W)}, cpp={sum(e['lang']=='cpp' for e in W)})")

    _write(I, os.path.join(out_dir, "incomplete.json"))
    print(f"wrote {len(I)} tasks -> {out_dir}/incomplete.json  "
          f"(missing_condition={sum(e['sub_kind']=='missing_condition' for e in I)}, "
          f"partial_steps={sum(e['sub_kind']=='partial_steps' for e in I)})")


if __name__ == "__main__":
    main()
