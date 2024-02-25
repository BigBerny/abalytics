from abalytics import analyze_independent_groups, analyze_dependent_groups
from abalytics.utils import format_results_as_table
import pandas as pd
import numpy as np

# Example 1 DataFrame for demonstration
data = {
    "variable": [
        x * 1.2 if i < 500 else x for i, x in enumerate(np.random.rand(1000))
    ],  # 100 random numeric values, 1.2 times higher for group A
    "group": [
        "A" if x < 500 else "B" for x in range(1000)
    ],  # 50 'A's followed by 50 'B's
}

df = pd.DataFrame(data)

result = analyze_independent_groups(
    df,
    "variable",
    "group",
    p_value_threshold=0.3,
)
print(format_results_as_table([result], identifiers_list=[{"Test name": "Example 1"}]))


# Example 2 DataFrame for demonstration
data = {
    "variable": [
        True if x < 40 or (50 <= x < 70) else False for x in range(100)
    ],  # Significantly more True values in group A than in group B
    "group": [
        "A" if x < 50 else "B" for x in range(100)
    ],  # 50 'A's followed by 50 'B's
}
df = pd.DataFrame(data)

result = analyze_independent_groups(
    df,
    "variable",
    "group",
    p_value_threshold=0.3,
)
print(format_results_as_table([result], identifiers_list=[{"Test name": "Example 2"}]))


# Example 3 DataFrame for pair-wise comparison
data = {
    "variable1": np.random.rand(1000),  # 100 random numeric values
    "variable2": np.random.rand(1000) * 1.1,  # 100 random numeric values
}
df = pd.DataFrame(data)

print("A/B test example")

result = analyze_dependent_groups(df, ["variable1", "variable2"])
print(format_results_as_table([result], identifiers_list=[{"Test name": "Example 3"}]))
