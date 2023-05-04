import pandas as pd
import matplotlib.pyplot as plt
import itertools


def CountProducts(df):
    
    products = [col for col in df.columns if col.startswith("product_")]
    
    df['date'] = pd.to_datetime(df['date'])
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    #IF THE dfFRAME IS NOT ORDERED BY DATE:
    # df = df.sort_values(by='date')
    
    print("Grouping by year, month and customer code...")
    grouped_df = df.groupby(['year', 'month', 'customer_code']).sum().reset_index()
    
    print("Computing differences...")
    grouped_df.sort_values(by=['customer_code', 'year', 'month'], inplace=True)
    grouped_df.set_index(['year', 'month', 'customer_code'], inplace=True)
    
    print("Calculating buys and cancels...")
    buyed_df = grouped_df.groupby(level='customer_code').diff().reset_index()
    cancelled_df = grouped_df.groupby(level='customer_code').diff().reset_index()
    
    print("Counting buys and cancels...")
    buyed_df[products] = buyed_df[products].applymap(lambda x: 1 if x == 1 else 0)
    cancelled_df[products] = cancelled_df[products].applymap(lambda x: 1 if x == -1 else 0)
    
    print("Grouping by year and month, and counting the product purchase...")
    monthly_added_counts = buyed_df.groupby(['year', 'month'])[products].sum().reset_index()
    monthly_cancelled_counts = cancelled_df.groupby(['year', 'month'])[products].sum().reset_index()
    
    print("Done!")
    return monthly_added_counts, monthly_cancelled_counts


def BarPlotForCountProducts(df):
    
    products = [col for col in df.columns if col.startswith("product_")]
    
    print("Plotting products count...")
    colors = itertools.cycle(['steelblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])
    
    # Convert the date column to a datetime object and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    

    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Loop over each product and plot the counts
    for product in products:
        # Filter the data to include only rows where the product equals to 1
        df_product = df[df[product] == 1]
        # Group the filtered data by year and month and count the occurrences
        product_counts = df_product.groupby([pd.Grouper(freq='M')])[product].sum()
        # Plot the counts on the subplot
        product_counts.plot(kind='bar', ax=ax, label=product, alpha=0.5, color=next(colors))

    # Set the axis labels and title
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Counts')
    ax.set_title('Product Counts')

    # Add a legend to the plot
    ax.legend()

    # Show the figure
    plt.show()
    
    # Group the data by year and month and count the occurrences of each product
    product_counts_by_month = df.groupby([pd.Grouper(freq='M')])[products].sum()

    # Display the top 9 products for each month in the terminal
    for month, data in product_counts_by_month.iterrows():
        top_products = data.nlargest(9)
        print(f"\nTop products for {month.strftime('%B %Y')}:")
        for i, product in enumerate(top_products.index):
            print(f"{i+1}. {product}: {top_products[product]}")


def LinearPlotForCountProducts(df):
    # Convert the 'year' and 'month' columns to a datetime format
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
    
    # Drop the 'year' and 'month' columns
    df.drop(['year', 'month'], axis=1, inplace=True)
    
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)
    
    # Plot the data using pandas plotting function
    df.plot(figsize=(10, 6))
    
    # Set plot title and axes labels
    plt.title('Product Performance over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    

    
    # Show the plot
    plt.show()

