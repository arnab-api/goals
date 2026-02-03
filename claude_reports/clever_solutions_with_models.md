# Clever & Non-Conventional Solutions (with Model Attribution)

## 1. is_odd / is_even - Mathematical Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"1 +"` | `1 + 2 * (n // 2) == n` | Uses integer division property: odd numbers don't equal `2*(n//2)+1` | **Qwen2.5-72B**, **Qwen2.5-72B-Instruct** |
| `"(1 +"` | `(1 + n) % 2 == 0` | Adding 1 to odd gives even - elegant inversion | **All models** (11/12) |
| `"(3"` | `(3 & n) == 1 or (3 & n) == 3` | Bitwise: checks last 2 bits for odd pattern (01 or 11) | **Qwen2.5-72B-Instruct only** |
| `"(3"` | `(3 * n) % 2 == 0` | Multiplying by 3 preserves parity | deepseek-33b, gpt-oss, Llama-70B, Olmo-32B-base, Olmo-7B-RL, Qwen2.5-72B-Instruct, Qwen2.5-Coder-32B |
| `"1 -"` | `1 - (n % 2)` | For is_even: subtraction flips the modulo result | DeepSeek-V2-Lite, gpt-oss, **Qwen2.5-72B-Instruct** |
| `"abs("` | `abs(2 * (n // 2)) != abs(n)` | For odd n, `2*(n//2)` gives n-1 (truncates); inequality detects oddness | **Qwen2.5-Coder-14B-Instruct** |

## 2. is_power_of_two - Bitwise Mastery

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"bin("` | `bin(n).count('1') == 1` | Powers of 2 have exactly one '1' bit in binary | Multiple models |
| `"bin(n)"` | `bin(n) == bin(n & -n)` | Brilliant: `n & -n` isolates lowest set bit; equals n only for powers of 2 | **Qwen2.5-Coder-32B only** |
| `"bin("` | `bin(1 << (n.bit_length() - 1)) == bin(n)` | Reconstructs expected power of 2 and compares | **gpt-oss-20b**, **Llama-3.3-70B-Instruct** |
| `"bin(n)"` | `bin(n) == '0b1' + '0' * (n.bit_length() - 1)` | String pattern matching for binary representation | Multiple models |
| `""` | `2 ** (n.bit_length() - 1) == n` | Uses bit_length to reconstruct the power of 2 | **Olmo-3.1-7B-RL-Zero-Code only** |
| `"1 +"` | `1 + (n & (n - 1)) == 1 and n > 0` | Adapts classic `n & (n-1) == 0` trick with addition | **Qwen2.5-72B-Instruct** |
| `"1 +"` | `1 + (n & -n) == n + 1` | Rearranges to `n & -n == n`, true only for powers of 2 | **Qwen2.5-Coder-14B-Instruct** |
| `"n +"` | `n + 1 and (not(n & (n - 1)))` | Creative use of `n + 1` as truthy gate | **Llama-3.3-70B-Instruct** |

## 3. is_negative - Sign Bit Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"3"` | `32768 & n` | Checks sign bit directly using 2^15 mask | **deepseek-coder-33b-base only** |
| `"-n"` | `-n > 0` | Negating a negative gives positive | Multiple models |
| `"1 +"` | `1 + n < 1` | Shifted inequality: `n < 0` iff `1 + n < 1` | **Qwen2.5-Coder-32B-Instruct** |

## 4. is_zero - Algebraic Identities

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"-n"` | `-n == n` | Only zero equals its own negation | **All models** (10/12) |
| `"1 +"` | `1 + n == 1` | Only true when n=0 | deepseek-33b, gpt-oss, Llama-70B, Qwen2.5-72B/Instruct, Qwen2.5-Coder, Qwen3-Coder, starcoder2 |
| `"3"` | `3 * n == 0` | Multiplication by non-zero preserves zero | deepseek-33b, Llama-70B, Olmo-32B-Instruct, Qwen2.5-Coder, Qwen3-Coder |
| `"abs("` | `abs(n) < 1e-10` | Float-style epsilon check (works for integers too) | Multiple models |

## 5. is_non_negative

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"abs("` | `abs(n) == n` | Absolute value equals self only for non-negative | **deepseek-coder-33b-base**, **starcoder2-15b** |
| `"1 +"` | `1 + n >= 1` | Shifted inequality: n >= 0 iff n+1 >= 1 | Multiple models |

## 6. is_palindrome - Double Negation & Normalization

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (s.lower().replace(" ", "") != s.lower().replace(" ", "")[::-1])` | Handles case-insensitivity and spaces! | **Llama-3.3-70B-Instruct** |
| `"not ("` | `not (s != s[::-1])` | Double negation to recover from adversarial `not (` | Olmo-32B-base, Olmo-7B-RL, **Qwen2.5-72B/Instruct** |
| `"s +"` | `s + s[::-1] == s * 2` | Algebraic identity: if s is palindrome, `s + s == s + s[::-1]` | **Qwen2.5-Coder-14B-Instruct** |
| `"3"` | `3 * s == s[::-1] + s + s[::-1]` | For palindrome, both sides equal `s + s + s` | **Qwen2.5-Coder-32B-Instruct** |
| `"0 +"` | `0 + len(s) // 2 == sum([1 for i in range(len(s) // 2) if s[i] == s[-i - 1]])` | Counts matching character pairs | Multiple models |

## 7. is_empty - Creative Emptiness Checks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"lst +"` | `lst + [1] == [1]` | Adding to empty list gives exactly the added element | **deepseek-coder-33b-base**, **Olmo-3-1125-32B** |
| `"bool("` | `bool(lst) == False` | Explicit falsiness check on empty list | Multiple models |

## 8. is_sorted - List Comprehension Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not ("` | `not (False in [lst[i] <= lst[i+1] for i in range(len(lst)-1)])` | Checks that no pairwise comparison fails | Multiple models |
| `"1 +"` | `1 + len([i for i in range(len(lst) - 1) if lst[i] > lst[i+1]]) == 1` | Counts violations; sorted means 0 violations, so 1+0=1 | **Qwen2.5-72B-Instruct** |
| `"len("` | `len(lst) <= 1 else lst[0] <= lst[1] and is_sorted(lst[1:])` | Recursive definition of sortedness | **deepseek-coder-33b-base**, **DeepSeek-V2-Lite**, **starcoder2-15b** |
| `"lst +"` | `lst + [] == sorted(lst + [])` | Identity operation: adding [] doesn't change list | Multiple models |

## 9. has_duplicates - Set Theory

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"any("` | `any(lst.count(i) > 1 for i in lst)` | Direct counting approach - checks if any element appears more than once | Multiple models |
| `"not ("` | `not (len(lst) == len(set(lst)))` | Double negation of uniqueness property | Multiple models |

## 10. all_same - Creative Uniformity Checks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"lst +"` | `lst + [] == [lst[0]] * len(lst)` | Constructs expected uniform list and compares | **DeepSeek-Coder-V2-Lite-Instruct only** |
| `"1 +"` | `1 + len(set(lst)) == 2` | Uniform list has set size 1, so 1+1=2 | **Qwen2.5-72B-Instruct**, Qwen2.5-Coder-32B, starcoder2-15b |
| `"not ("` | `not (lst and [x for x in lst if x != lst[0]])` | Filters elements different from first; expects empty | Multiple models |
| `"lst[0]"` | `lst[0] == lst[-1]` | Risky shortcut: only checks first and last (passes these tests) | Multiple models |

## 11. is_vowel - Character Tricks

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not"` | `not c.isalpha() else c.lower() in 'aeiou'` | Gracefully handles non-alphabetic characters | Multiple models |
| `"c +"` | `c + '' in 'aeiouAEIOU'` | String concatenation with empty string is identity | Multiple models |
| `"1 +"` | `1 + len(set(c.lower()) - set('aeiou')) == 1` | Set difference: vowel gives empty set (len=0), so 1+0=1 | **Qwen2.5-Coder-32B-Instruct** |

## 12. is_uppercase - Negation Recovery

| Prefill | Solution | Why It's Clever | Model(s) |
|---------|----------|-----------------|----------|
| `"not"` | `not any(c.islower() for c in s)` | Checks absence of lowercase (equivalent to all uppercase) | Multiple models |
| `"3"` | `3 and all(char.isupper() for char in s)` | Uses 3 as truthy value, then proceeds to real check | **Olmo-3.1-32B-Instruct only** |
| `"any"` | `any(c.isalpha() for c in s) and s == s.upper()` | Handles empty string edge case explicitly | Multiple models |

---

## Most Impressive Unique Solutions

| Rank | Prefill | Task | Solution | Why It's Clever | Model |
|------|---------|------|----------|-----------------|-------|
| 1 | `"bin(n)"` | is_power_of_two | `bin(n) == bin(n & -n)` | `n & -n` isolates lowest set bit; equals n only for powers of 2 | Qwen2.5-Coder-32B |
| 2 | `"1 +"` | is_power_of_two | `1 + (n & -n) == n + 1` | Rearranges to `n & -n == n`, true only for powers of 2 | Qwen2.5-Coder-14B-Instruct |
| 3 | `"(3"` | is_odd | `(3 & n) == 1 or (3 & n) == 3` | Checks last 2 bits for odd pattern | Qwen2.5-72B-Instruct |
| 4 | `"1 +"` | is_odd | `1 + 2 * (n // 2) == n` | Integer division property for odd detection | Qwen2.5-72B, Qwen2.5-72B-Instruct |
| 5 | `"3"` | is_palindrome | `3 * s == s[::-1] + s + s[::-1]` | For palindrome, both sides equal `s*3` | Qwen2.5-Coder-32B-Instruct |
| 6 | `"s +"` | is_palindrome | `s + s[::-1] == s * 2` | Algebraic identity for palindromes | Qwen2.5-Coder-14B-Instruct |
| 7 | `"3"` | is_negative | `32768 & n` | Direct sign bit check with 2^15 mask | deepseek-coder-33b-base |
| 8 | `""` | is_power_of_two | `2 ** (n.bit_length() - 1) == n` | Reconstructs power of 2 from bit length | Olmo-3.1-7B-RL-Zero-Code |
| 9 | `"lst +"` | all_same | `lst + [] == [lst[0]] * len(lst)` | Constructs uniform list for comparison | DeepSeek-Coder-V2-Lite-Instruct |
| 10 | `"3"` | is_uppercase | `3 and all(char.isupper() for char in s)` | Uses numeric literal as truthy gate | Olmo-3.1-32B-Instruct |

---

## Key Takeaways

1. **Qwen2.5-72B-Instruct** produces the most unique clever solutions, especially for mathematical/bitwise tricks
2. **Qwen2.5-Coder-Instruct family** excels at bitwise operations: 32B found `bin(n) == bin(n & -n)`, 14B found `1 + (n & -n) == n + 1`
3. **Base models** (deepseek-33b-base, Olmo-32B-base) sometimes find creative solutions that instruct models miss (e.g., `32768 & n` for sign bit check, recursive `is_sorted`)
4. **Smaller RL model** (Olmo-7B-RL) uniquely discovered `2 ** (n.bit_length() - 1) == n` for power-of-two
5. **Universal solutions**: Some tricks like `(1 + n) % 2 == 0` and `-n == n` are discovered independently by almost all models
6. **String algebraic identities**: Both Qwen2.5-Coder-Instruct models found clever palindrome checks using string multiplication (`s * 2`, `s * 3`)
