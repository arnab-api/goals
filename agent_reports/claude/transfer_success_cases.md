# Transfer success cases (full prompts) — for figure building

*From the scored 250-head runs. KV: `kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad`; Q: `diff_trigger/ph-goal_250__a2__piv-eq` (order_matched/order_diff, py→py); Exp B: `diff_trigger/ph-goal_250__a2__piv-eq__restrict/c_in_dst`. Q & Exp B prompts reconstructed via the seed-stable sampler.*

---

## Part 1 — KV-transfer cross-language cases (Figure-3-left style)


### datatype: `int`

#### py→cpp — SRC goal: *the absolute value* | DST native: *the last decimal digit*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is negative
(2) the absolute value
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = n < 0
    b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is negative
(2) the last decimal digit
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(int n){
    auto a = (n < 0);
    auto b =
```
- source (→SRC): `abs(n)
    return a, b
</code>`
- destination (native): `abs(n) % 10;
    return std::make_pair(a, b);
}
</code>`
- **patched**: `(n < 0) ? -n : n;
    return std::make_pair(a, b);
}
</code>`
- readout src / dst / **patched**: `I'm calculating the absolute value of the variable `x` using `abs(x)`.` / `I'm determining the last digit of the number by using the modulus oper…` / **`I'm calculating the absolute value of the variable `x` using `abs(x)`.`**

#### cpp→py — SRC goal: *whether the number reads the same forwards and backwards* | DST native: *whether the number is a multiple of three*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is odd
(2) whether the number reads the same forwards and backwards
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(int n){
    auto a = (n % 2 != 0);
    auto b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is odd
(2) whether the number is a multiple of three
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = n % 2 != 0
    b =
```
- source (→SRC): `([&](){
        auto s = to_string(n);
        return equal(s.begin(), s.begin()…`
- destination (native): `n % 3 == 0
    return a, b
</code>`
- **patched**: `str(n) == str(n)[::-1]
    return a, b
</code>`
- readout src / dst / **patched**: `I'm checking if the variable is equal to its reverse (converted to str…` / `I'm checking if the number is divisible by 3.` / **`I'm checking if the variable `x` is equal to its reverse (`str(x) == str(x)[::-1]`).`**


### datatype: `str`

#### py→cpp — SRC goal: *the length of the string* | DST native: *whether the first character is a letter*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of whitespace-separated words
(2) the number of space characters
(3) the length of the string
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = len(s.split())
    b = s.count(' ')
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given a string `s`, computes:
(1) the number of whitespace-separated words
(2) the number of space characters
(3) whether the first character is a letter
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(std::string s){
    auto a = 0;
    bool a_in = false;
    for(char c : s){ if(isspace((unsigned char)c)) a_in=false; else if(!a_in){ a_in=true; a++; } }
    auto b = 0;
    for(char c : s) if(c==' ') b++;
    auto c =
```
- source (→SRC): `len(s)
    return a, b, c
</code>`
- destination (native): `!s.empty() && isalpha((unsigned char)s[0]);
    return std::make_tuple(a, b, c);…`
- **patched**: `s.length();
    return std::make_tuple(a, b, c);
}
</code>`
- readout src / dst / **patched**: `I'm counting the number of characters in the string, including spaces …` / `I'm checking if the first character of the string is an alphabetic let…` / **`I'm counting the number of characters in the string, including spaces and special characte…`**

#### cpp→py — SRC goal: *whether the string is a palindrome* | DST native: *the number of digit characters*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given a string `s`, computes:
(1) the length of the string
(2) the number of vowels
(3) whether the string is a palindrome
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(std::string s){
    auto a = s.size();
    auto b = 0;
    for (char c : s) if (std::string("aeiou").find(c) != std::string::npos) b++;
    auto c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the length of the string
(2) the number of vowels
(3) the number of digit characters
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = len(s)
    b = sum(1 for c in s if c in 'aeiou')
    c =
```
- source (→SRC): `std::equal(s.begin(), s.begin() + s.size()/2, s.rbegin());
    return std::make_…`
- destination (native): `sum(1 for c in s if c.isdigit())
    return a, b, c
</code>`
- **patched**: `s == s[::-1]
    return a, b, c
</code>`
- readout src / dst / **patched**: `I'm checking if the string is equal to its reverse (ignoring case).` / `I'm counting digits in the string.` / **`I'm checking if the string is equal to its reverse (ignoring case).`**


### datatype: `list[int]`

#### cpp→py — SRC goal: *the number of negative values* | DST native: *the number of positive values*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given a list of integers `nums`, computes:
(1) whether any value is even
(2) the number of negative values
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(std::vector<int> nums){
    auto a = false;
    for (int x : nums) if (x % 2 == 0) a = true;
    auto b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) whether any value is even
(2) the number of positive values
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = any(x % 2 == 0 for x in nums)
    b =
```
- source (→SRC): `0;
    for (int x : nums) if (x < 0) b++;
    return std::make_pair(a, b);
}
</c…`
- destination (native): `sum(1 for x in nums if x > 0)
    return a, b
</code>`
- **patched**: `sum(x < 0 for x in nums)
    return a, b
</code>`
- readout src / dst / **patched**: `I'm counting how many elements in the list are negative.` / `Counting non-negative integers.` / **`I'm counting how many numbers in the list are negative.`**


### datatype: `list[str]`

#### cpp→py — SRC goal: *the length of the longest word* | DST native: *the number of words shorter than three characters*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a C++ program that implements a function `func` which, given a list of strings `words`, computes:
(1) whether all words have distinct lengths
(2) the number of words written entirely in lowercase letters
(3) the length of the longest word
Assume that all the libraries have been already imported.
<|im_end|>
<|im_start|>assistant
<code>
auto func(std::vector<std::string> words){
    std::set<int> a_s;
    for(auto &a_w : words) a_s.insert((int)a_w.size());
    auto a = a_s.size() == words.size();
    auto b = 0;
    for(auto &b_w : words){ bool b_has = false, b_ok = true; for(char b_c : b_w){ if(std::isalpha((unsigned char)b_c)){ b_has = true; if(std::isupper((unsigned char)b_c)) b_ok = false; } } if(b_has && b_ok) b++; }
    auto c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) whether all words have distinct lengths
(2) the number of words written entirely in lowercase letters
(3) the number of words shorter than three characters
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = len(set(len(w) for w in words)) == len(words)
    b = sum(1 for w in words if w.islower())
    c =
```
- source (→SRC): `0;
    for(auto &c_w : words) c = std::max(c, (int)c_w.size());
    return std::…`
- destination (native): `sum(1 for w in words if len(w) < 3)
    return a, b, c
</code>`
- **patched**: `max(len(w) for w in words) if words else 0
    return a, b, c
</code>`
- readout src / dst / **patched**: `I'm determining the length of the longest string in the list.` / `I'm counting lines that have fewer than 3 characters.` / **`I'm determining the length of the longest string in the list using `max(len(word) for word…`**


---

## Part 2 — Query-transfer cases (Figure-4-left style), py→py

*Native destination writes B; the query patch makes it write D (the progress-position task). `order_matched` src=[a,b,c], `order_diff` src=[a,c,b]; dst=[a,b,d].*


### variant: `order_matched`

#### `int` — D (target): *the absolute value* | native B: *the last decimal digit* | src C: *whether the number is a power of two*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number reads the same forwards and backwards
(2) the last decimal digit
(3) whether the number is a power of two
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = str(abs(n)) == str(abs(n))[::-1]
    b = abs(n) % 10
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number reads the same forwards and backwards
(2) the last decimal digit
(3) the absolute value
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = str(abs(n)) == str(abs(n))[::-1]
    b =
```
- destination (native→B): `abs(n) % 10
    c = abs(n)
    return a, b, c
</code>`
- **patched (→D)**: `abs(n)
    return a, b
</code>`
- readout **patched**: `I'm evaluating the absolute value of the number at this line, which is `abs(number)`.`

#### `str` — D (target): *whether every character is a letter or a digit* | native B: *the number of alphabetic characters* | src C: *whether the string is a palindrome*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of space characters
(2) the number of alphabetic characters
(3) whether the string is a palindrome
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = s.count(' ')
    b = sum(1 for c in s if c.isalpha())
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of space characters
(2) the number of alphabetic characters
(3) whether every character is a letter or a digit
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = s.count(' ')
    b =
```
- destination (native→B): `sum(c.isalpha() for c in s)
    c = all(c.isalnum() for c in s)
    return a, b,…`
- **patched (→D)**: `s.isalpha()
    c = s.isalnum()
    return a, b, c
</code>`
- readout **patched**: `I'm checking if all characters in the string are alphanumeric using `str.isalnum()`.`

#### `list[int]` — D (target): *the largest value* | native B: *whether any value is zero* | src C: *whether any value appears more than once*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the integer mean of the values
(2) whether any value is zero
(3) whether any value appears more than once
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = sum(nums) // len(nums)
    b = 0 in nums
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the integer mean of the values
(2) whether any value is zero
(3) the largest value
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = sum(nums) // len(nums)
    b =
```
- destination (native→B): `any(x == 0 for x in nums)
    c = max(nums)
    return a, b, c
</code>`
- **patched (→D)**: `any(x == max(nums) for x in nums)
    c = max(nums)
    return a, b, c
</code>`
- readout **patched**: `I'm checking if any element in the list is greater than zero using `max(lst) > 0`.`

#### `list[str]` — D (target): *the number of words longer than five characters* | native B: *the number of distinct words* | src C: *the number of words that start with a vowel*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) whether any word is empty
(2) the number of distinct words
(3) the number of words that start with a vowel
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = any(len(w) == 0 for w in words)
    b = len(set(words))
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) whether any word is empty
(2) the number of distinct words
(3) the number of words longer than five characters
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = any(len(w) == 0 for w in words)
    b =
```
- destination (native→B): `len(set(words))
    c = sum(1 for w in words if len(w) > 5)
    return a, b, c
<…`
- **patched (→D)**: `sum(1 for w in words if len(w) > 5)
    c = len(set(words))
    return a, b, c
<…`
- readout **patched**: `I'm counting the number of unique elements in the list.`


### variant: `order_diff`

#### `int` — D (target): *whether the number is even* | native B: *the last decimal digit* | src C: *whether the number is prime*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) the number of digits
(2) whether the number is prime
(3) the last decimal digit
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = len(str(abs(n)))
    c = abs(n) % 10
    b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) the number of digits
(2) the last decimal digit
(3) whether the number is even
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = len(str(abs(n)))
    b =
```
- destination (native→B): `abs(n) % 10
    c = n % 2 == 0
    return a, b, c
</code>`
- **patched (→D)**: `n % 2 == 0
    return a, b
</code>`
- readout **patched**: `I'm checking if the number is even by using the modulus operator `%`. If `number % 2 == 0`, the number is even…`

#### `str` — D (target): *whether the string is a palindrome* | native B: *the length of the string* | src C: *the number of digit characters*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of lowercase letters
(2) the number of digit characters
(3) the length of the string
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = sum(1 for c in s if c.islower())
    c = len(s)
    b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of lowercase letters
(2) the length of the string
(3) whether the string is a palindrome
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = sum(1 for c in s if c.islower())
    b =
```
- destination (native→B): `len(s)
    c = s == s[::-1]
    return a, b, c
</code>`
- **patched (→D)**: `len(s)
    c = s == s[::-1]
    return a, b, c
</code>`
- readout **patched**: `I'm checking the length of the string, which is `len(s)`.`

#### `list[int]` — D (target): *the difference between the maximum and minimum values* | native B: *the sum of all values* | src C: *whether the values are sorted in non-decreasing order*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the number of odd values
(2) whether the values are sorted in non-decreasing order
(3) the sum of all values
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = sum(1 for x in nums if x % 2 != 0)
    c = sum(nums)
    b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the number of odd values
(2) the sum of all values
(3) the difference between the maximum and minimum values
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = sum(1 for x in nums if x % 2 != 0)
    b =
```
- destination (native→B): `sum(nums)
    c = max(nums) - min(nums)
    return a, b, c
</code>`
- **patched (→D)**: `max(nums) - min(nums)
    return b, a
</code>`
- readout **patched**: `I'm identifying the maximum and minimum values in the list, then calculating their difference.`

#### `list[str]` — D (target): *the number of words written entirely in lowercase letters* | native B: *whether any word is empty* | src C: *the number of words shorter than three characters*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of words that end with a consonant
(2) the number of words shorter than three characters
(3) whether any word is empty
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = sum(1 for w in words if w and w[-1].isalpha() and w[-1].lower() not in "aeiou")
    c = any(len(w) == 0 for w in words)
    b =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of words that end with a consonant
(2) whether any word is empty
(3) the number of words written entirely in lowercase letters
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = sum(1 for w in words if w and w[-1].isalpha() and w[-1].lower() not in "aeiou")
    b =
```
- destination (native→B): `any(not w for w in words)
    c = sum(1 for w in words if w.islower())
    retur…`
- **patched (→D)**: `any(1 for w in words if w.islower())
    c = sum(1 for w in words if w.islower()…`
- readout **patched**: `I'm checking if any element in the list is empty or if there's an empty string in the list.`


---

## Part 3 — Experiment B: content (C) cases (Appendix H.3), py→py

*Both prompts list all four sub-tasks; src=[a,b,c,d] prefill{a,b}, dst=[a,b,d,c] prefill{a}. Patched run writes the source's specific target C (not native B, not progress D).*


### datatype: `int`

#### C (content): *whether the number has exactly two digits* | native B: *whether the number is positive* | progress D: *whether the number is a perfect square*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is odd
(2) whether the number is positive
(3) whether the number has exactly two digits
(4) whether the number is a perfect square
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = n % 2 != 0
    b = (n > 0)
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) whether the number is odd
(2) whether the number is positive
(3) whether the number is a perfect square
(4) whether the number has exactly two digits
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = n % 2 != 0
    b =
```
- destination (native→B): `n > 0
    c = int(n**0.5)**2 == n
    d = 10 <= n <= 99
    return a, b, c, d
</code>`
- **patched (→C)**: `10
    t = 99
    10 <= number <= 99
    number >= 10 and number <= 99
    10 <= number and number <= 99
    10 <= number <= 99 and 99 >= nu…`
- readout **patched**: `I'm checking if the number is between 10 and 99 (inclusive).`

#### C (content): *whether the number is odd* | native B: *the last decimal digit* | progress D: *whether the number is prime*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) the number of set bits in the binary representation
(2) the last decimal digit
(3) whether the number is odd
(4) whether the number is prime
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = bin(abs(n)).count("1")
    b = abs(n) % 10
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given an integer `n`, computes:
(1) the number of set bits in the binary representation
(2) the last decimal digit
(3) whether the number is prime
(4) whether the number is odd
<|im_end|>
<|im_start|>assistant
<code>
def func(n: int):
    a = bin(abs(n)).count("1")
    b =
```
- destination (native→B): `abs(n) % 10
    c = all(n % i != 0 for i in range(2, int(n**0.5) + 1)) if n > 1 else False…`
- **patched (→C)**: `(p % 2 == 1)
    print(is)
    print(p % 2 == 1)
    print(p % 2 == 1)
    print(p % 2 == 1)
    print(p % 2 == 1)
    print(p % 2 == 1)
   …`
- readout **patched**: `I'm checking if the number is odd by using the modulus operator (`%`). If `number % 1` equals 1, it's odd.`


### datatype: `str`

#### C (content): *whether the string has letters and all of them are lowercase* | native B: *the length of the string* | progress D: *the number of space characters*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) whether every character is a letter or a digit
(2) the length of the string
(3) whether the string has letters and all of them are lowercase
(4) the number of space characters
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = s.isalnum()
    b = len(s)
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) whether every character is a letter or a digit
(2) the length of the string
(3) the number of space characters
(4) whether the string has letters and all of them are lowercase
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = s.isalnum()
    b =
```
- destination (native→B): `len(s)
    c = s.count(' ')
    d = s.isalpha() and s.islower()
    return a, b, c, d
</co…`
- **patched (→C)**: `string.ascii_lowercase
    if a.isalpha() and a.islower():
        return True
    else:
        return False`
- readout **patched**: `I'm checking if all characters in the string are alphabetic and if there is at least one alphabetic character.`

#### C (content): *whether the string is a palindrome* | native B: *whether the string has letters and all of them are lowercase* | progress D: *the last character*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of digit characters
(2) whether the string has letters and all of them are lowercase
(3) whether the string is a palindrome
(4) the last character
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = sum(1 for c in s if c.isdigit())
    b = s.islower()
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a string `s`, computes:
(1) the number of digit characters
(2) whether the string has letters and all of them are lowercase
(3) the last character
(4) whether the string is a palindrome
<|im_end|>
<|im_start|>assistant
<code>
def func(s: str):
    a = sum(1 for c in s if c.isdigit())
    b =
```
- destination (native→B): `s.islower() and s.isalpha()
    c = s[-1] if s else None
    d = s == s[::-1]
    return a…`
- **patched (→C)**: `s == s[::-1]
    print(s, s[::-1])
    print(s, s[::-1])
    print(s, s[::-1])
    print(s, s[::-1])
    print(s, s[::-1])
    print(s, s[::…`
- readout **patched**: `I'm checking if the current string is equal to its reverse (ignoring case).`


### datatype: `list[int]`

#### C (content): *whether the values are sorted in non-decreasing order* | native B: *the number of positive values* | progress D: *the product of all values*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the number of distinct values
(2) the number of positive values
(3) whether the values are sorted in non-decreasing order
(4) the product of all values
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = len(set(nums))
    b = sum(1 for x in nums if x > 0)
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) the number of distinct values
(2) the number of positive values
(3) the product of all values
(4) whether the values are sorted in non-decreasing order
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = len(set(nums))
    b =
```
- destination (native→B): `sum(1 for x in nums if x > 0)
    c = 1
    for x in nums:
        c *= x
    d = all(nums…`
- **patched (→C)**: `order non-decreasing
  (a <= b) = true
  (a > b) = false
  (a <= b) = (a <= b) or (a = b)
  (a <= b) = (a < b) or (a = b)
  (a <= b) = (a < …`
- readout **patched**: `I'm checking if the current element is greater than or equal to the previous one, indicating the list is sorted in non-decreasing order.`

#### C (content): *whether the values are sorted in non-decreasing order* | native B: *the number of odd values* | progress D: *the difference between the maximum and minimum values*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) whether any value is zero
(2) the number of odd values
(3) whether the values are sorted in non-decreasing order
(4) the difference between the maximum and minimum values
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = 0 in nums
    b = sum(1 for x in nums if x % 2 != 0)
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of integers `nums`, computes:
(1) whether any value is zero
(2) the number of odd values
(3) the difference between the maximum and minimum values
(4) whether the values are sorted in non-decreasing order
<|im_end|>
<|im_start|>assistant
<code>
def func(nums: list[int]):
    a = 0 in nums
    b =
```
- destination (native→B): `sum(1 for x in nums if x % 2 != 0)
    c = max(nums) - min(nums)
    d = all(nums[i] <= nu…`
- **patched (→C)**: `all(x <= y for x, y in zip(values, values[1:]))
    c = max(values) - min(values)
    d = sum(1 for x in values if x % 2 != 0)
    return a,…`
- readout **patched**: `I'm checking if the list is sorted by comparing each element with its next neighbor to ensure no element is greater than the one following i…`


### datatype: `list[str]`

#### C (content): *the number of words that contain a digit* | native B: *the number of words that start with a vowel* | progress D: *the length of the shortest word*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of distinct words
(2) the number of words that start with a vowel
(3) the number of words that contain a digit
(4) the length of the shortest word
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = len(set(words))
    b = sum(1 for w in words if w and w[0].lower() in "aeiou")
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of distinct words
(2) the number of words that start with a vowel
(3) the length of the shortest word
(4) the number of words that contain a digit
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = len(set(words))
    b =
```
- destination (native→B): `sum(1 for word in words if word[0].lower() in 'aeiou')
    c = min(len(word) for word in w…`
- **patched (→C)**: `0
  for i in words:
    if i.isdigit():
      word += 1
  print(word)
  print(len(set(words)))
  print(len(min(words, key=len)))
  print(sum…`
- readout **patched**: `I'm checking if any character in the string is a digit.`

#### C (content): *the total number of characters across all words* | native B: *whether any word is a palindrome* | progress D: *the total number of vowels across all words*

**SRC prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of words shorter than three characters
(2) whether any word is a palindrome
(3) the total number of characters across all words
(4) the total number of vowels across all words
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = sum(1 for w in words if len(w) < 3)
    b = any(w == w[::-1] for w in words)
    c =
```
**DST prompt:**
```
<|im_start|>system
You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.<|im_end|>
<|im_start|>user
Write a Python program that implements a function `func` which, given a list of strings `words`, computes:
(1) the number of words shorter than three characters
(2) whether any word is a palindrome
(3) the total number of vowels across all words
(4) the total number of characters across all words
<|im_end|>
<|im_start|>assistant
<code>
def func(words: list[str]):
    a = sum(1 for w in words if len(w) < 3)
    b =
```
- destination (native→B): `any(w == w[::-1] for w in words)
    c = sum(w.count(v) for w in words for v in 'aeiouAEIO…`
- **patched (→C)**: `sum(len(w) for w in words)
    return a, b, c
    return a, b, c, d
    return a, b, c, d
    return a, b, c, d, e
    return a, b, c, d, e,…`
- readout **patched**: `I'm counting the total number of characters in the string.`
