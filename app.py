import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

st.set_page_config(page_title= "Data Exploration Tool",
                   page_icon= "",
                   initial_sidebar_state= "collapsed",
                   layout="wide")

def convert_df_to_csv(data_frame):
    return data_frame.to_csv(index=False).encode('utf-8')

def display_numerical_analysis(df):
    st.subheader("Numerical Feature Analysis")
    numerical_cols = df.select_dtypes(include=['number'])
    if not numerical_cols.empty:
        st.dataframe(numerical_cols.describe().T)
        # You can add more specific numerical analysis here, like histograms,
        # box plots, correlation matrices, etc.
        st.subheader("Numerical Feature Visualizations")
        for col in numerical_cols.columns:
            st.write(f"**{col}**")
            # Example: Histogram
            # st.subheader(f"Distribution of {col}")
            # st.hist_chart(numerical_cols[col])
            # Example: Box Plot
            st.subheader(f"Box Plot of {col}")
            st.vega_lite_chart(numerical_cols, {
                'mark': {'type': 'boxplot'},
                'encoding': {
                    'y': {'field': col, 'type': 'quantitative'}
                }
            })
    else:
        st.info("No numerical features found in the DataFrame.")

def load_data():
    """Loads CSV File in to Pandas Data Frame"""
    uploaded_file = st.file_uploader(label= "Upload your csv file", type= ["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully")
        return df
    return None

def display_data_preview(df: DataFrame):
    """Displays a preview of the DataFrame."""
    st.subheader("Data Preview")
    st.dataframe(df.head())

def display_data_info(df: DataFrame):
    """Displays basic information about the DataFrame."""
    st.subheader("Data Information")
    d_types = [str(d) for d in df.dtypes]
    data_info = pd.DataFrame({"Columns": df.columns, "Data Type": d_types})
    st.dataframe(data_info)
    st.info(f"There are Number **{df.shape[0]}** rows and **{df.shape[1]}** columns")

def display_descriptive_info(df: DataFrame):
    col_num_stats, col_cat_stats = st.columns(2)
    with col_num_stats:
        st.subheader("Numerical Statistics")
        numerical_columns = df.select_dtypes(include=["number"])
        st.dataframe(numerical_columns.describe().T)
    with col_cat_stats:
        st.subheader("Categorical Statistics")
        categorical_columns = df.select_dtypes(include=["object", "category"])
        st.dataframe(categorical_columns.describe().T)

def display_data_cleaning(df: DataFrame):
    """Displays and handles inconsistent formatting, non-numeric values, missing values, and duplicate rows using tabs."""
    st.subheader("Data Cleaning")
    formatting_tab, missing_value_tab, duplicate_rows_tab = st.tabs(["Handle Formatting",
                                                                                      "Handle Missing Values",
                                                                                      "Handle Duplicate Rows"])
    with formatting_tab:
        st.subheader("White Space Handling")
        string_cols = df.select_dtypes(include=["object", "category"]).columns
        if not string_cols.empty:
            col_to_strip = st.multiselect("Select columns to trim whitespace", string_cols)
            if st.button("Trim Whitespace"):
                for col in col_to_strip:
                    df[col] = df[col].str.strip()
                st.success(f"Whitespace trimmed from columns: {', '.join(col_to_strip)}")
                st.session_state.df = df
                with st.expander("Preview Data after trimming"):
                    st.dataframe(df.head())
        else:
            st.info("No string columns found to trim whitespace.")

        st.subheader("Case Consistency")
        if not string_cols.empty:
            case_option = st.selectbox("Select case convertion option:",
                                       ["Do Nothing", "Lowercase", "Uppercase"])
            col_to_case = st.multiselect("Select columns to convert case", string_cols)
            if st.button("Convert Case"):
                if case_option == "Lowercase":
                    for col in col_to_case:
                        df[col] = df[col].str.lower()
                    st.success(f"Converted {col} to lowercase.")
                elif case_option == "Lowercase":
                    for col in col_to_case:
                        df[col] = df[col].str.upper()
                    st.success(f"Converted {col} to uppercase.")
                st.session_state.df = df
                with st.expander("Preview Data after case convertion"):
                    st.dataframe(df.head())

        else:
            st.info("No string columns found for case conversion.")


    # with non_numeric_tab:
    #     st.subheader("Non Numeric Data Handling")
    #     non_numeric_values = {}
    #     cols_to_check = df.select_dtypes(include=['number', 'object']).columns
    #     for col in cols_to_check:
    #         # Attempt to convert to numeric, coercing errors to NaN
    #         numeric_converted = pd.to_numeric(df[col], errors='coerce')
    #         # Identify rows where conversion failed but the original value was not already NaN
    #         non_numeric_mask = numeric_converted.isna() & df[col].notna()
    #         non_numeric_series = df[col][non_numeric_mask]  # Select the original column using the mask
    #         if not non_numeric_series.empty:
    #             unique_non_numeric = non_numeric_series.unique().tolist()
    #             non_numeric_values[col] = unique_non_numeric
    #
    #     if non_numeric_values:
    #         st.warning("Potential non-numeric values found in columns:")
    #         for col, values in non_numeric_values.items():
    #             st.write(f"- Column '{col}': {values}")

    with missing_value_tab:
        st.subheader("Missing Value Handling")
        total_missing = df.isnull().sum().sort_values(ascending=False)
        percent_missing = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)

        missing_info = pd.concat([total_missing, percent_missing],
                                 axis=1,
                                 keys=['Total Missing', 'Percent Missing'])
        missing_info = missing_info[missing_info['Total Missing'] > 0]

        if not missing_info.empty:
            st.warning(f"There are explicit missing values in dataframe.")
            with st.expander("Preview missing values"):
                st.dataframe(missing_info)
            handle_explicit_missing = st.selectbox(
                "What would you like to do with these missing values?",
                ["Do Nothing", "Remove Rows", "Fill Values"]
            )
            if handle_explicit_missing == "Remove Rows":
                original_rows = df.shape[0]
                df.dropna(inplace= True)
                st.success(f"Removed **{original_rows - df.shape[0]}** rows with missing values. New Shape: {df.shape}")
                st.session_state.df = df
                with st.expander("Preview Data after dropping NaN rows"):
                    st.dataframe(df.head())
            elif handle_explicit_missing == "Fill Values":
                strategy = st.selectbox("Select a strategy to fill missing values:",
                                        ["Mean", "Median", "Most Frequent", "Custom Value"])
                column_with_missing = missing_info.index.tolist()
                selected_columns_fill = st.multiselect("Select column to fill", column_with_missing)
                if strategy == "Mean":
                    for col in selected_columns_fill:
                        if df[col].dtype in ["int64", 'float64']:
                            df[col].fillna(value= df[col].mean(), inplace= True)
                            st.success(f"Missing value in {col} is filled with mean.")
                        else:
                            st.warning("Cannot fill non-numerical column with mean.")
                elif strategy == "Median":
                    for col in selected_columns_fill:
                        if df[col].dtype in ["int64", 'float64']:
                            df[col].fillna(value= df[col].median(), inplace= True)
                            st.success(f"Missing value in {col} is filled with median.")
                        else:
                            st.warning("Cannot fill non-numerical column with median.")
                elif strategy == "Most Frequent":
                    for col in selected_columns_fill:
                        try:
                            df[col].fillna(value = df[col].mode()[0], inplace= True)
                            st.success(f"Missing value in {col} is filled with most frequent value.")
                        except IndexError:
                            st.warning(f"Cannot fill {col} as there is no mode(all values are unique)")
                elif strategy == "Custom Value":
                    custom_value = st.text_input("Enter the value yo fill missing entries")
                    if custom_value is not None:
                        for col in selected_columns_fill:
                            df[col].fillna(value = custom_value, inplace= True)
                            st.success(f"Missing value in {col} is filled with {custom_value}.")
                    else:
                        st.warning(f"Cannot fill {col} as no values entered")
                if handle_explicit_missing == "Fill Values":
                    with st.expander("Updated Data Preview (after filling explicit NaNs)"):
                        st.session_state.df = df
                        st.dataframe(df.head())
        else:
            st.info(f"There are no missing values in dataset.")

    with duplicate_rows_tab:
        duplicate_rows_count = df.duplicated().sum()
        if duplicate_rows_count > 0:
            st.warning(f"There are **{duplicate_rows_count}** duplicate rows.")

            with st.expander("Preview Duplicate Rows"):
                st.dataframe(df[df.duplicated(keep="first")])

            handle_duplicates = st.selectbox(
                "What would you like to do with the duplicate rows?",
                ["Do Nothing", "Remove Duplicates"]
            )
            if handle_duplicates == "Remove Duplicates":
                original_rows = df.shape[0]
                df.drop_duplicates(inplace= True)
                st.success(f"Removed **{original_rows - df.shape[0]}** duplicate rows. New Shape: {df.shape}")
                st.session_state.df = df
                with st.expander("Preview Data after removing duplicate rows"):
                    st.dataframe(df.head())
        else:
            st.info("No duplicate rows found in the dataset.")

def display_visualizations(df: DataFrame):
    st.subheader("Data Visualization and Insights")
    st.subheader("Uni-variate Analysis")
    numerical_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    col1, col2 = st.columns(2)
    with col1:
        if not numerical_columns.empty:
            st.subheader("Numerical Columns")
            selected_numerical_column = st.selectbox("Select a numerical column for uni-variate analysis", numerical_columns)
            if selected_numerical_column:
                data = df[selected_numerical_column].dropna()
                st.write(f"**Distribution Analysis for `{selected_numerical_column}`:**")
                plot_type = st.selectbox("Select Plot Type",
                                       ["Histogram", "Box Plot", "Violin Plot"])
                if plot_type == "Histogram":
                    show_kde = st.checkbox("Show KDE", False)
                    fig, ax = plt.subplots()
                    sns.histplot(df, x=selected_numerical_column, kde=show_kde, ax = ax)
                    st.pyplot(fig)
                    st.write(f"**Insights:** The distribution of '{selected_numerical_column}' shows its frequency across different ranges. The KDE line provides an estimate of the probability density function.")
                elif plot_type == "Box Plot":
                    hide_fliers = st.checkbox("Hide Outliers", True)
                    fig, ax = plt.subplots()
                    sns.boxplot(df, x=selected_numerical_column, showfliers= hide_fliers ,ax = ax)
                    st.pyplot(fig)
                elif plot_type == "Violin Plot":
                    fig, ax = plt.subplots()
                    sns.violinplot(df, x=selected_numerical_column, ax = ax)
                    st.pyplot(fig)
        else:
            st.warning("No numerical columns available for univariate analysis.")
    with col2:
        if categorical_columns.any():
            st.subheader("Categorical Columns")
            selected_categorical_column_uni = st.selectbox("Select a categorical column for univariate analysis:",
                                                       categorical_columns)
            if selected_categorical_column_uni:
                unique_value_count = df[selected_categorical_column_uni].nunique()
                if unique_value_count > 10:
                    max_slider_value = unique_value_count
                    min_slider_value = 10
                    default_slider_value = min(15, max_slider_value)
                    top_n = st.slider("Number of top categories to display:",
                                      min_value=min_slider_value,
                                      max_value=max_slider_value,
                                      value=default_slider_value)
                    if max_slider_value > 50:
                        st.warning(
                            f"The column '{selected_categorical_column_uni}' has a very large number of unique values ({max_slider_value}). Consider using a higher 'Number of top categories to display' or other methods to visualize the distribution effectively.")
                else:
                    top_n = unique_value_count  # Display all if unique count is 10 or less
                top_categories = df[selected_categorical_column_uni].value_counts().nlargest(top_n)

                fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability
                sns.countplot(df, x=selected_categorical_column_uni , order=top_categories.index, ax=ax)
                plt.xticks(rotation=90)
                st.pyplot(fig)

                st.write("**Categorical Column Frequencies**")
                frequency_df = df[selected_categorical_column_uni].value_counts().reset_index()
                frequency_df.columns = ["Category", "Frequency"]
                st.dataframe(frequency_df.sort_values(by='Frequency', ascending=False))
        else:
            st.info("No categorical columns available for univariate analysis.")

def main():
    """Main function to run the Streamlit application."""
    st.title("CSV Data Analyzer")
    df = load_data()
    if df is not None:
        st.session_state.df = df.copy()  # Store the original DataFrame
        st.markdown("---")
        display_data_preview(st.session_state.df)
        st.markdown("---")
        display_data_info(st.session_state.df)
        st.markdown("---")
        display_descriptive_info(st.session_state.df)
        st.markdown("---")
        display_data_cleaning(st.session_state.df)
        st.markdown("---")
        display_visualizations(st.session_state.df)
        st.markdown("---")

if __name__ == "__main__":
    main()