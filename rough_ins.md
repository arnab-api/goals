Brainstorm with me: what is the best way to identify different scopes in a piece of code. Checkout the following code for example:

```
def bfs(graph, start):
    """Perform BFS on a graph starting from a given node."""
    visited = set()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend([node for node in graph[vertex] if node not in visited])
    return visited
```

It has different scopes:
1. The function scope of `bfs`
2. The loop scope of `while queue:`
3. The conditional scope of `if vertex not in visited:`
4. Each line can be its own scope as well. Or can be broken apart to multiple scopes. 
    * In the list comprehension `[node for node in graph[vertex] if node not in visited]`. There is a loop scope and a if scope.
    * Another example: `s.islower() and s == s[::-1]`, single line, but has two scopes with distinct goals. 

What is a good way to identify these scopes? With different levels of granularity?