import abalytics
import pandas as pd
import numpy as np

# Example DataFrame for demonstration
data = {
    "variable": np.random.rand(100),  # 100 random numeric values
    "group": [
        "A" if x < 50 else "B" for x in range(100)
    ],  # 50 'A's followed by 50 'B's
}
df = pd.DataFrame(data)

print("A/B test example")
print(
    abalytics.get_results_pretty_text(
        df,
        "variable",
        "group",
        identifiers=[f"A/B test example"],
        p_value_threshold=0.3,
    )
)

# Example DataFrame for demonstration
data = {
    "variable": np.random.choice(
        [True, False], size=100
    ),  # 100 random True or False values
    "group": [
        "A" if x < 50 else "B" for x in range(100)
    ],  # 50 'A's followed by 50 'B's
}
df = pd.DataFrame(data)

print("A/B test example")
print(
    abalytics.get_results_pretty_text(
        df,
        "variable",
        "group",
        identifiers=[f"A/B test example"],
        p_value_threshold=0.3,
    )
)

# Example DataFrame for pair-wise comparison
data = {
    "variable1": np.random.rand(1000),  # 100 random numeric values
    "variable2": np.random.rand(1000)*4,  # 100 random numeric values
}
df = pd.DataFrame(data)

print("A/B test example")

result = abalytics.analyze_dependent_groups(
    df,
    ["variable1", "variable2"]
)
print(abalytics.utils.format_results_as_table([result], identifiers_list=[["Test Identifier"]]))
