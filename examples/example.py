import abalytics
import pandas as pd
import numpy as np

# Example 1 DataFrame for demonstration
data = {
    "variable": [x * 1.2 if i < 500 else x for i, x in enumerate(np.random.rand(1000))],  # 100 random numeric values, 1.2 times higher for group A
    "group": [
        "A" if x < 500 else "B" for x in range(1000)
    ],  # 50 'A's followed by 50 'B's
}

df = pd.DataFrame(data)

result = abalytics.analyze_independent_groups(
    df,
    "variable",
    "group",
    p_value_threshold=0.3,
)
print(
    abalytics.utils.format_results_as_table([result], identifiers_list=[["Test 1 B"]])
)


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

result = abalytics.analyze_independent_groups(
    df,
    "variable",
    "group",
    p_value_threshold=0.3,
)
print(
    abalytics.utils.format_results_as_table([result], identifiers_list=[["Test 2 B"]])
)


# Example DataFrame for pair-wise comparison
data = {
    "variable1": np.random.rand(1000),  # 100 random numeric values
    "variable2": np.random.rand(1000) * 1.1,  # 100 random numeric values
}
df = pd.DataFrame(data)

print("A/B test example")

result = abalytics.analyze_dependent_groups(df, ["variable1", "variable2"])
print(
    abalytics.utils.format_results_as_table(
        [result], identifiers_list=[["Test Identifier"]]
    )
)
