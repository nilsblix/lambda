# lambda

> Single file REPL for a Lambda Calculus interpreter.

## Syntax:

A *variable* is some sequence of characters. For example: `\x.y` denotes a
function where the argument is a variable `x` and returns a variable `y`.
Variables can be any length, and can overload semantic tokens by wrapping
the variable name in `@"..."`. Example:

```
x
var
anything
@"some variable"
@"\wow.\is this a variable?"
```

A *function* takes the form: `\x.y` where the `x` variable is the function's
argument, and `y` is the returned expression. Note that `y` doesn't have to be
a variable, which enables important functional concepts, such as currying.
Examples:

```
\x.x         ; x -> x (the id function)
\x.y         ; x -> y
\x.y.z       ; x, y -> z
\x.y.f.f x y ; When applied to two arguments, it creates a pair.
```

An *application* of a function takes the form: `f x` where `f` is the function
to apply the argument `x` to. If `f` is *named*, then no parentheses are
needed. If however `f` is a lambda, then parantheses are needed to know which
applications are ended where. Examples:
```
f x      ; f(x)
g x y    ; g(x, y). Note that `g` has to be defined as a curried function, i.e `g = \x.y.z`, thus making `(g x)` equal to `\y.z`.
(\x.x) y ; Lambdas are wrapped in parentheses during application.
```

When chaining multiple function applications, such as `fst (pair 12 13)`, then
parentheses are needed to denote precedence, as `fst pair 12 13` is ambiguous.

*Comments* are denoted via `;`. The parser ignores everything from `;` until a
newline.

## Examples

### Booleans

With this system, we can define simple types, such as booleans:

```
true = \x.y.x
false = \x.y.y
```

But, why does these two functions behave as booleans? Lets try to use them
in some if/else statement. But first lets introduce an `if` function.

```
if = \x.x ; (i.e the id function)
```

Lets see what happens when we evaluate the following expression:

```
if true then else         ; ==
((\x.x) \x.y.x) then else ; ==
(\x.y.x) then else        ; ==
(\y.then) else            ; ==
then
```

Wow, our entire expression got evaluated down to `then`. You can yourself
try it with `false`, and see that it gets reduced to `else`.

### Pairs

Let us now take the `pair` example from above, and try to evaluate it with
two arguments:

```
; Remember the definition of pair
pair = \x.y.f.f x y
pair 12 13 ; == (\f.f) 12 13
```

We can define `fst` and `snd` functions to extract these two items in our
pair:

```
fst = \p.p true
snd = \p.p false
```

Lets try them on our pair:

```
fst (pair 12 13)          ; ==
(\p.p true) (\f.f 12 13)  ; ==
(\f.f 12 13) true         ; ==
true 12 13                ; ==
12

snd (pair 12 13)          ; ==
(\p.p false) (\f.f 12 13) ; ==
(\f.f 12 13) false        ; ==
false 12 13               ; ==
13
```
