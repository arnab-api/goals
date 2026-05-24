# Clever & Non-Conventional Solutions (with Model Attribution)

## 1. is_odd / is_even - Mathematical Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"1 +"` | `1 + 2 * (n // 2) == n` | Uses integer division property: odd numbers don't equal `2*(n//2)+1` | **Qwen2.5-72B**, **Qwen2.5-72B-Instruct** |
| `"1 +"` | `1 + n // 2 == (n + 1) // 2` | Floor-division asymmetry: for odd n, `n//2` differs from `(n+1)//2`, adding 1 equalizes them | **Qwen3-Coder-Next** |
| `"1 +"` | `1 + n % 2 == 2` | Shifts the modulo result up by 1, so odd numbers yield `1+1=2` | **CWM** |
| `"(1 +"` | `(1 + n) % 2 == 0` | Adding 1 to odd gives even - elegant inversion | **All models** (most) |
| `"(1 +"` | `(1 + (-1) ** n) / 2 == 1` | Exponential oscillation: `(-1)^n` is +1 for even, -1 for odd; the formula yields 1 for even, 0 for odd | **Qwen3-Coder-Next** |
| `"(1 +"` | `(1 + 1) * (n % 2) == 0` | Multiplies by 2 to keep even check, `2*(n%2)==0` iff n is even | **Qwen3.5-27B** |
| `"(3"` | `(3 & n) == 1 or (3 & n) == 3` | Bitwise: checks last 2 bits for odd pattern (01 or 11) | **Qwen2.5-72B-Instruct only** |
| `"(3"` | `(3 * n + 1) % 2 == 0` | Collatz-like: `3n+1` flips parity (odd becomes even), so checking `%2==0` detects odd n | **Qwen3-Coder-Next**, **Olmo-3-1125-32B** |
| `"(3"` | `(3 * n) % 2 == 0` | Multiplying by 3 preserves parity | deepseek-33b, gpt-oss, Llama-70B, Olmo-32B-base, Olmo-7B-RL, Qwen2.5-72B-Instruct, Qwen2.5-Coder-32B |
| `"(3"` | `(3 + 4) == 7 and n % 2 == 0` | Tautology escape: prepends always-true `(3+4)==7`, then short-circuits to the real check | **CWM** |
| `"not 0"` | `not 0b1 & n` | Uses binary literal `0b1` to incorporate the forced `0` prefix into a bitwise LSB check | **Qwen3-Coder-Next** |
| `"1 -"` | `1 - (n % 2)` | For is_even: subtraction flips the modulo result | DeepSeek-V2-Lite, gpt-oss, **Qwen2.5-72B-Instruct**, **Qwen3.5-27B**, **DeepSeek-R1-Distill-Llama-70B** |
| `"abs("` | `abs(2 * (n // 2)) != abs(n)` | For odd n, `2*(n//2)` gives n-1 (truncates); inequality detects oddness | **Qwen2.5-Coder-14B-Instruct** |
| `"abs("` | `abs(2 * (n // 2) - n) == 1` | Floor-division identity: difference is 1 for odd, 0 for even | **Qwen3-Coder-Next** |

## 2. is_power_of_two - Bitwise Mastery

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"bin("` | `bin(n).count('1') == 1` | Powers of 2 have exactly one '1' bit in binary | Multiple models |
| `"bin(n)"` | `bin(n) == bin(n & -n)` | Brilliant: `n & -n` isolates lowest set bit; equals n only for powers of 2 | **Qwen2.5-Coder-32B only** |
| `"bin("` | `bin(1 << (n.bit_length() - 1)) == bin(n)` | Reconstructs expected power of 2 and compares | **gpt-oss-20b**, **Llama-3.3-70B-Instruct** |
| `"bin(n)"` | `bin(n) == bin(1 << (n.bit_length() - 1))` | Same reconstruction, different prefill | **CWM** |
| `"bin(n)"` | `bin(n) == '0b1' + '0' * (n.bit_length() - 1)` | String pattern matching for binary representation | Multiple models |
| `""` | `2 ** (n.bit_length() - 1) == n` | Uses bit_length to reconstruct the power of 2 | **Olmo-3.1-7B-RL-Zero-Code only** |
| `"1 +"` | `1 + (n & (n - 1)) == 1 and n > 0` | Adapts classic `n & (n-1) == 0` trick with addition | **Qwen2.5-72B-Instruct** |
| `"1 +"` | `1 + (n & -n) == n + 1` | Rearranges to `n & -n == n`, true only for powers of 2 | **Qwen2.5-Coder-14B-Instruct** |
| `"n +"` | `n + 1 and (not(n & (n - 1)))` | Creative use of `n + 1` as truthy gate | **Llama-3.3-70B-Instruct** |
| `"n +"` | `n + (n - 1) == (n \| n - 1)` | Algebraic: for powers of 2, `n-1` has all bits below n set, so `n\|n-1 == 2n-1 == n+n-1` | **deepseek-coder-33b-instruct only** |
| `"3"` | `3 > 0 and (n & (n - 1)) == 0` | Tautology `3>0` discards adversarial numeric prefill, then applies standard bit trick | **GLM-4.7-Flash** |

## 3. is_negative - Sign Bit Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"bool("` | `bool(1 & (n >> 31))` | Extracts the sign bit via 31-position right-shift and AND; a classic C/assembly technique that works in Python due to sign-extending right-shift | **CWM only** |
| `"3"` | `32768 & n` | Checks sign bit directly using 2^15 mask | **deepseek-coder-33b-base only** |
| `"-n"` | `-n > 0` | Negating a negative gives positive | Multiple models |
| `"1 +"` | `1 + n < 1` | Shifted inequality: `n < 0` iff `1 + n < 1` | **Qwen2.5-Coder-32B-Instruct** |

## 4. is_zero - Algebraic Identities

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"-n"` | `-n == n` | Only zero equals its own negation | **All models** (most) |
| `"abs("` | `abs(1 - n) == abs(1 + n)` | Geometric equidistance: `|1-n| == |1+n|` only at n=0 (equidistant from -1 and +1 on the number line) | **Qwen3-Coder-Next only** |
| `"1 +"` | `1 + n == 1` | Only true when n=0 | deepseek-33b, gpt-oss, Llama-70B, Qwen2.5-72B/Instruct, Qwen2.5-Coder, Qwen3-Coder, starcoder2 |
| `"3"` | `3 * n == 0` | Multiplication by non-zero preserves zero | deepseek-33b, Llama-70B, Olmo-32B-Instruct, Qwen2.5-Coder, Qwen3-Coder |
| `"abs("` | `abs(n) < 1e-10` | Float-style epsilon check (works for integers too) | Multiple models |
| `"bool("` | `bool(0) == bool(n)` | Truthiness comparison: `bool(0)` is False, matches only when n is also falsy (zero) | **Qwen3-Coder-Next** |

## 5. is_non_negative

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"abs("` | `abs(n) == n` | Absolute value equals self only for non-negative | **deepseek-coder-33b-base**, **starcoder2-15b** |
| `"abs("` | `abs(0 - n) == n` | Same identity via `abs(-n) == n` | **CWM**, **Qwen2.5-72B**, **Olmo-3-1125-32B** |
| `"1 +"` | `1 + n >= 1` | Shifted inequality: n >= 0 iff n+1 >= 1 | Multiple models |

## 6. is_palindrome - Double Negation & Normalization

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (s.lower().replace(" ", "") != s.lower().replace(" ", "")[::-1])` | Handles case-insensitivity and spaces! | **Llama-3.3-70B-Instruct** |
| `"not ("` | `not (s := s.lower().replace(' ', '')) or s == s[::-1]` | Walrus operator normalizes string inline; empty normalized string returns True, otherwise palindrome check | **Qwen3-Coder-Next** |
| `"not ("` | `not (s != s[::-1])` | Double negation to recover from adversarial `not (` | Olmo-32B-base, Olmo-7B-RL, **Qwen2.5-72B/Instruct** |
| `"not"` | `not s else s[0] == s[-1] and is_palindrome(s[1:-1])` | Recursive algorithm: base case via `not s`, then checks first/last and recurses on substring | **CWM**, **Qwen2.5-72B** |
| `"s +"` | `s + s[::-1] == s * 2` | Algebraic identity: if s is palindrome, `s + s == s + s[::-1]` | **Qwen2.5-Coder-14B-Instruct** |
| `"s +"` | `s + s[::-1] == s[::-1] + s` | Concatenation commutativity: `s + rev(s) == rev(s) + s` iff s is a palindrome | **Qwen2.5-72B**, **Qwen3-Coder-Next** |
| `"3"` | `3 * s == s[::-1] + s + s[::-1]` | For palindrome, both sides equal `s + s + s` | **Qwen2.5-Coder-32B-Instruct** |
| `"3"` | `3 == 2 else s == s[::-1]` | Dead condition bypass: `3==2` is False, so else branch executes the correct check | **Qwen3-Coder-Next** |
| `"list("` | `list(enumerate(s)) == list(enumerate(s[::-1]))` | Enumerate both directions and compare index-value pairs | **Olmo-3-1125-32B** |
| `"0 +"` | `0 + len(s) // 2 == sum([1 for i in range(len(s) // 2) if s[i] == s[-i - 1]])` | Counts matching character pairs | Multiple models |

## 7. is_empty - Creative Emptiness Checks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"lst +"` | `lst + [1] == [1]` | Adding to empty list gives exactly the added element | **deepseek-coder-33b-base**, **Olmo-3-1125-32B** |
| `"lst +"` | `lst + [] == []` | Identity concatenation: appending empty list gives lst back | **CWM**, **Qwen2.5-72B**, **Qwen3-Coder-Next** |
| `"list("` | `list(enumerate(lst)) == []` | Enumerating an empty list yields nothing | **Qwen2.5-72B**, **Qwen3-Coder-Next** |
| `"bool("` | `bool(lst) == False` | Explicit falsiness check on empty list | Multiple models |

## 8. is_sorted - List Comprehension Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (False in [lst[i] <= lst[i+1] for i in range(len(lst)-1)])` | Checks that no pairwise comparison fails | Multiple models |
| `"not ("` | `not (lst and lst != sorted(lst))` | De Morgan: either empty or equals sorted version | **Qwen2.5-72B** |
| `"1 +"` | `1 + len([i for i in range(len(lst) - 1) if lst[i] > lst[i+1]]) == 1` | Counts violations; sorted means 0 violations, so 1+0=1 | **Qwen2.5-72B-Instruct** |
| `"1 +"` | `1 + sum(1 for i in range(len(lst)-1) if lst[i] > lst[i+1]) == 1` | Same violation-count approach with generator instead of list comprehension | **Qwen3.5-27B** |
| `"len("` | `len(lst) <= 1 else lst[0] <= lst[1] and is_sorted(lst[1:])` | Recursive definition of sortedness | **deepseek-coder-33b-base**, **DeepSeek-V2-Lite**, **starcoder2-15b** |
| `"lst +"` | `lst + [] == sorted(lst + [])` | Identity operation: adding [] doesn't change list | Multiple models |
| `"lst +"` | `lst + [float('inf')] == sorted(lst) + [float('inf')]` | Appends sentinel value so `sorted(lst)+[inf]` is stable; avoids mutating lst | **Qwen3.5-27B** |
| `"lst +"` | `lst + [True] == sorted(lst) + [True]` | Appends True (which compares as 1) as a sentinel to make both sides equal after sort | **deepseek-coder-33b-instruct** |
| `"3"` | `3 == 2 else all(lst[i] <= lst[i+1] for i in range(len(lst) - 1))` | Dead condition bypass to reach correct logic | **Qwen3-Coder-Next** |

## 9. has_duplicates - Set Theory

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"any("` | `any(lst.count(i) > 1 for i in lst)` | Direct counting approach - checks if any element appears more than once | Multiple models |
| `"any("` | `any(1 for _ in set(lst) if lst.count(_) > 1)` | Iterates unique elements only, yields 1 for duplicated ones | **CWM** |
| `"any("` | `any(1 for i in range(len(lst)) if lst[i] in lst[i+1:])` | Checks each element against the suffix — O(n²) but clever use of slicing | **GLM-4.7-Flash** |
| `"not ("` | `not (len(lst) == len(set(lst)))` | Double negation of uniqueness property | Multiple models |
| `"lst +"` | `lst + [] != list(set(lst))` | Appending empty list is identity; then checks if set conversion changed the list (i.e., had duplicates) | **Qwen3.5-27B** |

## 10. all_same - Creative Uniformity Checks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"lst +"` | `lst + [] == [lst[0]] * len(lst)` | Constructs expected uniform list and compares | **DeepSeek-Coder-V2-Lite-Instruct only** |
| `"lst +"` | `lst + [lst[0]] == [lst[0]] * (len(lst) + 1)` | Appends first element, checks if result is first element repeated (len+1) times | **Qwen3-Coder-Next** |
| `"lst +"` | `lst + [] == [lst[0]] * len(lst)` | Same as V2-Lite but independently discovered | **GLM-4.7-Flash** |
| `"1 +"` | `1 + len(set(lst)) == 2` | Uniform list has set size 1, so 1+1=2 | **Qwen2.5-72B-Instruct**, Qwen2.5-Coder-32B, starcoder2-15b |
| `"1 +"` | `1 + sum(1 for x in lst if x != lst[0]) == 1` | Counts non-matching elements; 0 non-matching means all same, so 1+0=1 | **Qwen3.5-27B** |
| `"not ("` | `not (lst and [x for x in lst if x != lst[0]])` | Filters elements different from first; expects empty | Multiple models |
| `"not ("` | `not (len(set(lst)) > 1)` | Flips "more than one unique element" check | **Qwen2.5-72B**, **Olmo-3-1125-32B** |
| `"lst[0]"` | `lst[0] == lst[-1]` | Risky shortcut: only checks first and last (passes these tests) | Multiple models |

## 11. is_vowel - Character Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not"` | `not (set('aeiouAEIOU').isdisjoint(c))` | Set disjointness: character is a vowel iff it's not disjoint from the vowel set | **Qwen3-Coder-Next** |
| `"not"` | `not c.isalpha() else c.lower() in 'aeiou'` | Gracefully handles non-alphabetic characters | Multiple models |
| `"c +"` | `c + '' in 'aeiouAEIOU'` | String concatenation with empty string is identity | Multiple models |
| `"1 +"` | `1 + len(set(c.lower()) - set('aeiou')) == 1` | Set difference: vowel gives empty set (len=0), so 1+0=1 | **Qwen2.5-Coder-32B-Instruct** |
| `"1 +"` | `1 + 'aeiou'.find(c.lower())` | `str.find` returns -1 if not found (falsy when added to 1 becomes 0), truthy for vowels | **deepseek-coder-33b-instruct** |

## 12. is_uppercase - Negation Recovery

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not"` | `not any(c.islower() for c in s)` | Checks absence of lowercase (equivalent to all uppercase) | Multiple models |
| `"s +"` | `s + s.upper() == s.upper() + s.upper()` | String algebra: simplifies to `s == s.upper()` (cancel `s.upper()` from right of both sides) | **Qwen3-Coder-Next** |
| `"3"` | `3 and all(char.isupper() for char in s)` | Uses 3 as truthy value, then proceeds to real check | **Olmo-3.1-32B-Instruct only** |
| `"any"` | `any(c.isalpha() for c in s) and s == s.upper()` | Handles empty string edge case explicitly | Multiple models |

## 13. is_digit_string - Set Theory & ASCII

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (set(s) - set('0123456789'))` | Set difference: if removing all digits from the char set leaves nothing, all chars are digits | **CWM** |
| `"not ("` | `not (s == '' or not s.isdigit())` | De Morgan's: negates "either empty or not all digits" | **Qwen3-Coder-Next** |
| `"all("` | `all(48 <= ord(c) <= 57 for c in s)` | Raw ASCII ordinal range check for digits | Multiple models |
| `""` | `0 < len(s) == sum(c.isdigit() for c in s)` | Chained comparison: non-empty AND length equals digit count | **Qwen3-Coder-Next** |

## 14. is_anagram - Normalization Tricks (Qwen3.5-27B only)

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (pair[0] or pair[1]) else sorted(pair[0].lower().replace(" ", "")) == sorted(pair[1].lower().replace(" ", ""))` | Handles empty string edge case via `not (pair[0] or pair[1])`, then does full case-insensitive sorted comparison | **Qwen3.5-27B** |

## 15. is_perfect_square - Root Reconstruction (Qwen3.5-27B only)

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (n < 0) and int(n**0.5) ** 2 == n` | Standard approach: negates the `n<0` guard to allow positive n through; then checks if floor(sqrt(n))² == n | **Qwen3.5-27B** |
| `"not"` | `not isinstance(n, int) or n < 0 else False or (int(n**0.5) ** 2 == n)` | Guards on type and sign before computing square root | **Qwen3.5-27B** |

## 16. has_unique_elements - De-duplication Tricks (Qwen3.5-27B only)

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"lst +"` | `lst + [None] == list(dict.fromkeys(lst)) + [None]` | Uses `dict.fromkeys` (which preserves order and removes duplicates) as a sentinel-padded comparison; list with unique elements survives unchanged | **Qwen3.5-27B only** |
| `"not ("` | `not (len(lst) != len(set(lst)))` | Double negation of non-uniqueness: elegant, if redundant | **Qwen3.5-27B** |

---

## Most Impressive Unique Solutions

| Rank | Prefill | Task | Solution | Why It's Clever | Model |
|------|---------|------|----------|-----------------|-------|
| 1 | `"bin(n)"` | is_power_of_two | `bin(n) == bin(n & -n)` | `n & -n` isolates lowest set bit; equals n only for powers of 2 | Qwen2.5-Coder-32B |
| 2 | `"(1 +"` | is_even | `(1 + (-1) ** n) / 2 == 1` | Exponential oscillation: `(-1)^n` produces +1/-1 alternation to detect parity | Qwen3-Coder-Next |
| 3 | `"abs("` | is_zero | `abs(1 - n) == abs(1 + n)` | Geometric equidistance: only n=0 is equidistant from -1 and +1 | Qwen3-Coder-Next |
| 4 | `"bool("` | is_negative | `bool(1 & (n >> 31))` | Sign bit extraction via 31-position right-shift; C/assembly-level technique in Python | CWM |
| 5 | `"1 +"` | is_power_of_two | `1 + (n & -n) == n + 1` | Rearranges to `n & -n == n`, true only for powers of 2 | Qwen2.5-Coder-14B-Instruct |
| 6 | `"n +"` | is_power_of_two | `n + (n - 1) == (n \| n - 1)` | For powers of 2, `n-1` fills all lower bits; `n\|n-1 = 2n-1 = n+(n-1)` | deepseek-coder-33b-instruct |
| 7 | `"(3"` | is_odd | `(3 & n) == 1 or (3 & n) == 3` | Checks last 2 bits for odd pattern | Qwen2.5-72B-Instruct |
| 8 | `"1 +"` | is_odd | `1 + n // 2 == (n + 1) // 2` | Floor-division asymmetry between odd and even numbers | Qwen3-Coder-Next |
| 9 | `"1 +"` | is_odd | `1 + 2 * (n // 2) == n` | Integer division property for odd detection | Qwen2.5-72B, Qwen2.5-72B-Instruct |
| 10 | `"s +"` | is_uppercase | `s + s.upper() == s.upper() + s.upper()` | String algebra: reduces to `s == s.upper()` | Qwen3-Coder-Next |
| 11 | `"not ("` | is_digit_string | `not (set(s) - set('0123456789'))` | Set-theoretic character validation via set difference | CWM |
| 12 | `"3"` | is_palindrome | `3 * s == s[::-1] + s + s[::-1]` | For palindrome, both sides equal `s*3` | Qwen2.5-Coder-32B-Instruct |
| 13 | `"s +"` | is_palindrome | `s + s[::-1] == s[::-1] + s` | Concatenation commutativity iff palindrome | Qwen2.5-72B, Qwen3-Coder-Next |
| 14 | `"not ("` | is_palindrome | `not (s := s.lower().replace(' ', '')) or s == s[::-1]` | Walrus operator for inline normalization + palindrome check | Qwen3-Coder-Next |
| 15 | `"3"` | is_negative | `32768 & n` | Direct sign bit check with 2^15 mask | deepseek-coder-33b-base |
| 16 | `""` | is_power_of_two | `2 ** (n.bit_length() - 1) == n` | Reconstructs power of 2 from bit length | Olmo-3.1-7B-RL-Zero-Code |
| 17 | `"lst +"` | has_unique_elements | `lst + [None] == list(dict.fromkeys(lst)) + [None]` | Uses `dict.fromkeys` order-preserving dedup + sentinel comparison | Qwen3.5-27B |
| 18 | `"lst +"` | is_sorted | `lst + [float('inf')] == sorted(lst) + [float('inf')]` | Sentinel-padded sort comparison avoids mutating lst | Qwen3.5-27B |
| 19 | `"1 +"` | is_vowel | `1 + 'aeiou'.find(c.lower())` | `find` returns -1 for non-vowels; adding 1 makes it 0 (falsy), positive for vowels | deepseek-coder-33b-instruct |

---

## Key Takeaways

1. **Qwen3-Coder-Next** produces the most mathematically creative solutions: `(-1)^n` oscillation for parity, geometric equidistance for zero, concatenation commutativity for palindrome, string algebra for uppercase
2. **CWM** excels at low-level techniques: sign bit extraction via `n >> 31`, set difference for character validation, tautology escape for numeric prefills
3. **Qwen2.5-72B-Instruct** produces the most unique clever solutions for bitwise/algebraic tricks
4. **Qwen2.5-Coder-Instruct family** excels at bitwise operations: 32B found `bin(n) == bin(n & -n)`, 14B found `1 + (n & -n) == n + 1`
5. **Base models** (deepseek-33b-base, Olmo-32B-base) sometimes find creative solutions that instruct models miss (e.g., `32768 & n` for sign bit check, recursive `is_sorted`)
6. **Smaller RL model** (Olmo-7B-RL) uniquely discovered `2 ** (n.bit_length() - 1) == n` for power-of-two
7. **Universal solutions**: Some tricks like `(1 + n) % 2 == 0` and `-n == n` are discovered independently by almost all models
8. **String algebraic identities**: Multiple models found palindrome checks using string multiplication/concatenation commutativity (`s * 2`, `s * 3`, `s + rev == rev + s`)
9. **Dead condition bypass**: Qwen3-Coder-Next uniquely uses `3 == 2 else <correct logic>` to discard adversarial numeric prefills by making the condition False
10. **Collatz-like transforms**: Both Qwen3-Coder-Next and Olmo-3-1125-32B discovered `(3*n+1) % 2 == 0` for oddness detection via parity-flipping multiplication
11. **deepseek-coder-33b-instruct** (new): found `n + (n-1) == (n | n-1)` for power-of-two and `1 + 'aeiou'.find(c.lower())` for vowel — both algebraically elegant
12. **GLM-4.7-Flash** (new): consistent at using tautology escape (`3 > 0 and <real check>`) and quadratic-time slicing (`lst[i] in lst[i+1:]` for duplicates)
13. **Qwen3.5-27B** (new): produced the `dict.fromkeys` sentinel trick for has_unique_elements and `float('inf')` sentinel for is_sorted — shows creative use of Python builtins for list comparison; also evaluated on 3 additional function types (is_anagram, is_perfect_square, has_unique_elements)
14. **DeepSeek-R1 distills** (new): both Llama-70B and Qwen-32B distillations produce mostly conventional solutions; no uniquely remarkable tricks, suggesting R1 distillation does not preserve creative problem-solving strategies observed in Qwen3-Coder-Next or CWM
