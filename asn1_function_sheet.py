
import pandas as pd
import numpy as np
import math

def age_splitter(df, col_name, age_threshold):
    """
    Splits the dataframe into two dataframes based on an age threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_name (str): The name of the column containing age values.
    age_threshold (int): The age threshold for splitting.

    Returns:
    tuple: A tuple containing two dataframes:
        - df_below: DataFrame with rows where age is below the threshold.
        - df_above_equal: DataFrame with rows where age is above or equal to the threshold.
    """
    df_below = df[df[col_name] < age_threshold]
    df_above_equal = df[df[col_name] >= age_threshold]
    return df_below, df_above_equal
    
def effectSizer(df, num_col, cat_col):
    """
    Calculates the effect sizes of binary categorical classes on a numerical value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    num_col (str): The name of the numerical column.
    cat_col (str): The name of the binary categorical column.

    Returns:
    float: Cohen's d effect size between the two groups defined by the categorical column.
    Raises:
    ValueError: If the categorical column does not have exactly two unique values.
    """
    # Clean the categorical column
    df[cat_col] = df[cat_col].astype(str).str.strip()

    # Get unique values
    values = df[cat_col].dropna().unique()
    if len(values) != 2:
        raise ValueError(f"{cat_col} must have exactly 2 unique values, found: {values}")

    # Split numerical column into two groups
    g1 = df[df[cat_col] == values[0]][num_col]
    g2 = df[df[cat_col] == values[1]][num_col]

    # Calculate Cohen's d
    mean_diff = g1.mean() - g2.mean()
    pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)

    if pooled_std == 0 or np.isnan(pooled_std):
        return 0

    return mean_diff / pooled_std


def cohenEffectSize(group1, group2):
    # You need to implement this helper function
    # This should not be too hard...

    group1, group2 = group1.dropna(), group2.dropna()
    if group1.empty or group2.empty:
        return np.nan
    
    mean_diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt(((group1.std() ** 2) + (group2.std() ** 2)) / 2)
    
    if pooled_std == 0:
        return 0.0  
    
    return mean_diff / pooled_std


def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):
    """
    This function takes a dataframe and a list of cohort column names, and returns a dictionary
    where each key is a cohort name and each value is an object containing the specified statistics
    """
    results = {}

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    for cohort_col in cohorts:
        for cohort_value in df[cohort_col].unique():
            cohort_df = df[df[cohort_col] == cohort_value]
            metric = CohortMetric(f"{cohort_col}={cohort_value}")

            # Numeric stats
            stats_dict = {}
            for num_col in numeric_cols:
                series = cohort_df[num_col]
                stats_dict[num_col] = {}
                if 'mean' in statistics:
                    stats_dict[num_col]['mean'] = series.mean()
                if 'median' in statistics:
                    stats_dict[num_col]['median'] = series.median()
                if 'std' in statistics:
                    stats_dict[num_col]['std'] = series.std()
                if 'min' in statistics:
                    stats_dict[num_col]['min'] = series.min()
                if 'max' in statistics:
                    stats_dict[num_col]['max'] = series.max()
            metric.statistics["numeric_stats"] = stats_dict

            # Categorical counts
            counts = {}
            for cat_col in categorical_cols:
                counts[cat_col] = cohort_df[cat_col].value_counts().to_dict()
            metric.statistics["counts"] = counts

            results[f"{cohort_col}={cohort_value}"] = metric

    return results

class CohortMetric():
    # don't change this
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.statistics = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }
    def setMean(self, new_mean):
        self.statistics["mean"] = new_mean
    def setMedian(self, new_median):
        self.statistics["median"] = new_median
    def setStd(self, new_std):
        self.statistics["std"] = new_std
    def setMin(self, new_min):
        self.statistics["min"] = new_min
    def setMax(self, new_max):
        self.statistics["max"] = new_max

    def compare_to(self, other):
        for stat in self.statistics:
            if not self.statistics[stat].equals(other.statistics[stat]):
                return False
        return True
    def __str__(self):
        output_string = f"\nCohort: {self.cohort_name}\n"
        for stat, value in self.statistics.items():
            output_string += f"\t{stat}:\n{value}\n"
            output_string += "\n"
        return output_string
